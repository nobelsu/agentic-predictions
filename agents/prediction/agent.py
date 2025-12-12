from pydantic import BaseModel
import asyncio
import argparse
import os
from datetime import datetime, timezone, timedelta
import aiosqlite
import traceback

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.config import Settings, MCPSettings, MCPServerSettings, LoggerSettings, GoogleSettings
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# DO NOT CHANGE START

class PredictionResponse(BaseModel):
    prediction: bool
    reason: str 

# DO NOT CHANGE END

settings = Settings(
    name="prediction-agent",
    execution_engine="asyncio",
    logger=LoggerSettings(
        transports=["console", "file"],
        level="debug",
        progress_display=True,
        path_settings={
            "path_pattern":"logs/mcp-agent-{unique_id}.jsonl",
            "unique_id":"timestamp",
            "timestamp_format":"%Y%m%d_%H%M%S",
        },
    ),
    mcp=MCPSettings(
        servers={
            "sequential-thinking": MCPServerSettings(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
                env={"MAX_HISTORY_SIZE": "1000"},
            ),
            "crawl4ai": MCPServerSettings(
                transport="sse",
                url="http://localhost:11235/mcp/sse"
            )
        }
    ),
    google=GoogleSettings(default_model="gemini-2.5-flash"),
)

async def predictSuccess(prompts, actual):
    app = MCPApp(settings=settings)

    async with app.run() as agent_app:
        prediction_instruction = f"""
            You are an expert venture capital analyst agent. Your job is to predict whether a start-up will be an "outlier success" based on the founder's anonymised profile.
            
            You will be provided the data:
            1. industry: The industry which the start-up is operating in. 
            2. ipos: Previous IPOs by the founder 
            3. acquisitions: Previous acquisitions by the founder 
            4. educations_json: Educational background 
            5. jobs_json: Professional background 
            6. anonymised_prose: Narrative description.

            A start-up is considered successful if:
            - Exits via IPO at a valuation exceeding $500M;
            - Gets acquired for more than $500M;
            - Raises over $500M in total funding.

            Your task is to predict whether or not the startup will succeed. You will need to output:
            1. prediction: Whether or not the startup will succeed (True/False)
            2. reason: Reasoning for prediction
            
            Use sequential thinking to reason and formulate a plan.
            
            Use this crawl4ai perform deep research on reports or articles.
        """

        agent = Agent(
            name="prediction-agent",
            instruction=prediction_instruction,
            server_names=["crawl4ai","sequential-thinking"],
        )
        convertor_agent = Agent(
            name="convertor-agent",
            instruction="""
                You are a data convertor agent. 

                Your task is to generate a structured response from the unstructured output of another AI agent.

                Follow the response model provided:
                prediction: Boolean value indicating the agent's prediction (success or not)
                reason: String value indicating the agent's explanation for the prediction

                Use only the information you are fed. Set the field to be an empty string if you find nothing in the output matching it. Do not leave it as `None`.
            """,
            server_names=[]
        )

        async with agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            convertor_llm = await convertor_agent.attach_llm(GoogleAugmentedLLM)
# DO NOT CHANGE START
            results = []
            for prompt in prompts:
                try:
                    result = await llm.generate_str(
                        message=prompt,
                        request_params=RequestParams(
                            max_iterations=20
                        ),
                    )
                    
                    try:
                        converted_result = await convertor_llm.generate_structured(
                            message=result,
                            response_model=PredictionResponse, 
                        )
                        results.append(converted_result)
                    except Exception as e:
                        print(f"Error converting result: {e}")
                        traceback.print_exc()
                        # Fallback or skip
                        # Creating a dummy failure response to avoid crashing the whole batch
                        results.append(PredictionResponse(
                            prediction=False,
                            reason=f"Error during conversion: {str(e)}",
                        ))
                        
                except Exception as e:
                    print(f"Error generating prediction: {e}")
                    traceback.print_exc()
                    results.append(PredictionResponse(
                        prediction=False,
                        reason=f"Error during generation: {str(e)}",
                    ))

            blocks = []
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for i, r in enumerate(results, 1):
                if (actual[i-1] == "1"):
                    if (r.prediction):
                        tp += 1
                    else:
                        fn += 1
                else:
                    if (r.prediction):
                        fp += 1
                    else:
                        tn += 1
                block = [
                        f"Prediction {i}:",
                        f"Agent answer: {r.prediction}",
                        f"Correct answer: {str(actual[i-1] == "1")}",
                        f"Reasoning: {r.reason}",
                        ""
                    ]
                blocks.append("\n".join(block))
            precision = 0
            recall = 0
            accuracy = 0
            fscore = 0
            if (tp+fp):
                precision = tp/(tp+fp)
            if (tp+fn):
                recall = tp/(tp+fn)
            if (tp+fn+tn+fp):
                accuracy = (tp+tn)/(tp+fn+tn+fp)
            if (tp+fn+fp):
                fscore = (1.25*tp)/(1.25*tp+0.25*fn+fp)
            return (f"**REPORT OF RESULTS:**\n\nF_0.5 score: {fscore}\nPrecision: {precision}\nRecall: {recall}\nAccuracy: {accuracy}\n\n") + ("\n".join(blocks))

async def uploadReport(text):
    async with aiosqlite.connect("files.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)

        async with db.execute("SELECT COUNT(*) FROM reports") as cur:
            (count,) = await cur.fetchone()
        if count == 0:
            await db.execute("DELETE FROM sqlite_sequence WHERE name='reports'")
            await db.commit()

        await db.execute("""
            INSERT INTO reports (content, created_at)
            VALUES (?, ?)
        """, (text, datetime.now(timezone(timedelta(hours=7)))))
        await db.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction Agent script")
    parser.add_argument("--p",nargs="+",help="List of prompts to provide agent")
    parser.add_argument("--a",nargs="+",help="Actual results")

    args = parser.parse_args()
    prompts = args.p
    actual = args.a
    
    report = asyncio.run(predictSuccess(prompts, actual))
    asyncio.run(uploadReport(report))

# DO NOT CHANGE END
