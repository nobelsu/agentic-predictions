from pydantic import BaseModel
import asyncio
import argparse
import traceback
from datetime import datetime, timezone, timedelta
import aiosqlite
import csv

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
            "path_pattern":"logs/prediction/prediction-{unique_id}.jsonl",
            "unique_id":"session_id",
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
            ),
            "arxiv": MCPServerSettings(
                transport="sse",
                url="http://localhost:8050/sse"
            )
        }
    ),
    google=GoogleSettings(default_model="gemini-2.5-flash"),
)

async def predictSuccess(prompts, success_values):
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

            Use sequential thinking to reason about this.
            
            Use arxiv to find research papers on arxiv, then use crawl4ai perform deep research on reports or articles.

            Your task is to predict whether or not the startup will succeed. You will need to output:
            1. prediction: Whether or not the startup will succeed (True/False)
            2. reason: One paragraph reasoning for prediction

            DO NOT output anything else.
        """

        agent = Agent(
            name="prediction-agent",
            instruction=prediction_instruction,
            server_names=["crawl4ai","sequential-thinking","arxiv"],
        )
        convertor_agent = Agent(
            name="convertor-agent",
            instruction="""
                You are a data convertor agent. 

                Your task is to generate a structured response from the unstructured output of another AI agent.

                Follow the response model provided:
                prediction: Boolean value indicating the agent's prediction (success or not)
                reason: String value indicating the agent's explanation for the prediction

                Use only the information you are fed. 
            """,
            server_names=[]
        )

        async with agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            async with convertor_agent:
                convertor_llm = await convertor_agent.attach_llm(GoogleAugmentedLLM)
# DO NOT CHANGE START
                results = []
                for prompt in prompts:
                    try:
                        result = await llm.generate(
                            message=prompt,
                            request_params=RequestParams(
                                max_iterations=10
                            ),
                        )
                        
                        try:
                            converted_result = await convertor_llm.generate_structured(
                                message=str(result),
                                response_model=PredictionResponse, 
                            )
                            results.append(converted_result)
                        except Exception as e:
                            print(f"Error converting result: {e}")
                            traceback.print_exc()
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
                    if (success_values[i-1] == "1"):
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
                            f"Correct answer: {str(success_values[i-1] == "1")}",
                            f"Reasoning: {r.reason}",
                            ""
                        ]
                    blocks.append("\n".join(block))
                
                precision = "No data because no true or false positives"
                recall = "No data because no true positive or false negative"
                fscore = "No data because no true positive, false negative, or false positive"
                
                if (tp+fp):
                    precision = f"{tp/(tp+fp)}"
                if (tp+fn):
                    recall = f"{tp/(tp+fn)}"
                if (tp+fn+fp):
                    fscore = f"{(1.25*tp)/(1.25*tp+0.25*fn+fp)}"

                with open("values.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([tp,fp,tn,fn])
                
                return (f"**REPORT OF RESULTS:**\n\nF_0.5 score: {fscore}\nPrecision: {precision}\nRecall: {recall}\n\n") + ("\n".join(blocks))

async def upload(content, table_name="reports", database_name="files.db"):
    async with aiosqlite.connect(database_name) as db:
        await db.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)

        async with db.execute(f"SELECT COUNT(*) FROM {table_name}") as cur:
            (count,) = await cur.fetchone()
        if count == 0:
            await db.execute(f"DELETE FROM sqlite_sequence WHERE name='{table_name}'")
            await db.commit()

        await db.execute(f"""
            INSERT INTO {table_name} (content, created_at)
            VALUES (?, ?)
        """, (content, datetime.now(timezone(timedelta(hours=7)))))
        await db.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction Agent script")
    parser.add_argument("--p",nargs="+",help="List of prompts to provide agent")
    parser.add_argument("--s",nargs="+",help="Actual success values")

    args = parser.parse_args()
    prompts = args.p
    success_values = args.s
    
    report = asyncio.run(predictSuccess(prompts, success_values))
    asyncio.run(upload(report))

# DO NOT CHANGE END
