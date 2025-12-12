from pydantic import BaseModel
import asyncio
import argparse
from datetime import datetime, timezone, timedelta
import aiosqlite

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.config import Settings, MCPSettings, MCPServerSettings, LoggerSettings, OpenAISettings
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# DO NOT CHANGE START

class PredictionResponse(BaseModel):
    prediction: bool
    reason: str 
    thought_process: list[str] 
    tool_history: list[str]

# DO NOT CHANGE END

settings = Settings(
    name="prediction-agent",
    execution_engine="asyncio",
    logger=LoggerSettings(
        transports=["console", "file"],
        level="info",
        progress_display=True,
        path_settings={
            "path_pattern":"logs/mcp-agent-{unique_id}.jsonl",
            "unique_id":"timestamp",
            "timestamp_format":"%Y%m%d_%H%M%S",
        },
    ),
    mcp=MCPSettings(
        servers={
            "g-search": MCPServerSettings(command="npx", args=["-y", "g-search-mcp"]),
            "fetcher": MCPServerSettings(command="npx", args=["-y", "fetcher-mcp"]),
            "sequential-thinking": MCPServerSettings(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
                env={"MAX_HISTORY_SIZE": "1000"},
            ),
            "think-mcp": MCPServerSettings(
                command="uvx",
                args=["think-mcp"],
                env={"TAVILY_API_KEY":"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"}
            )
        }
    ),
    openai=OpenAISettings(default_model="gpt-5.1"),
)

async def predictSuccess(prompts, actual):
    app = MCPApp(settings=settings)

    async with app.run() as agent_app:
        prediction_instruction = f"""
            You are an agent whose job is to predict whether a start-up will be an outlier success, based on founder profiles.
            
            You will be given an anonymised founder's profile. You will be provided the data:
            1. industry: The industry which the start-up is operating in. 
            2. ipos: Previous IPOs by the founder
            3. acquisitions: Previous acquisitions by the founder
            4. educations_json: Educational background of the founder
            5. jobs_json: Professional background of the founder
            6. anonymised_prose: An anonymised paragraph containing the founder's educational and professional background and the industry of his startup.

            A start-up is considered successful if:
            - Exits via IPO at a valuation exceeding $500M;
            - Gets acquired for more than $500M;
            - Raises over $500M in total funding.

            Your task is to predict whether or not the founder will succeed. You will need to output:
            1. prediction: Whether or not the founder will succeed
            2. reason: Reason for prediction
            3. thought_process: Sequence of thoughts to arrive at your conclusion
            4. tool_history: Sequence of MCP servers and tools used to arrive at your conclusion. Be specific about what you obtained from using them. If you did not use the server, do not lie and return an empty list. 

            Use sequential thinking to reason about this.
            
            The other servers include: 
            - g-search: to search for relevant information that could help with reasoning.
            - fetcher: to read the websites found by search and extract the necessary data.
        """

        agent = Agent(
            name="prediction-agent",
            instruction=prediction_instruction,
            server_names=["g-search", "fetcher", "sequential-thinking"],
        )
        convertor_agent = Agent(
            name="convertor-agent",
            instruction="""
                You are a data convertor agent. 

                Your task is to simply structure the unstructured output of another AI agent into the format given.

                Use sequential thinking to reason about this.

                Think carefully. Do not make any assumptions. Do not lie. Use only the output you will be fed.
            """,
            server_names=["think-mcp","sequential-thinking"]
        )

        async with agent:
            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            convertor_llm = await convertor_agent.attach_llm(OpenAIAugmentedLLM)
# DO NOT CHANGE START
            results = []
            for prompt in prompts:
                result = await llm.generate_str(message=prompt,request_params=RequestParams(
                    max_iterations=20  # Set your desired limit
                ),)
                converted_result = await convertor_llm.generate_structured(message=result,response_model=PredictionResponse)
                results.append(converted_result)

            blocks = []
            cnt = 0
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
                        f"Thought process:\n{'\n'.join(r.thought_process) if r.thought_process else '-'}",
                        f"Tool history:\n{'\n'.join(r.tool_history) if r.tool_history else '-'}",
                        ""
                    ]
                blocks.append("\n".join(block))
            precision = f"{tp}/{tp+fp}"
            recall = f"{tp}/{tp+fn}"
            accuracy = f"{tp+tn}/{tp+fn+tn+tp}"
            return (f"**REPORT OF RESULTS:**\n\nPrecision: {precision}\nRecall: {recall}\nAccuracy: {accuracy}\n\n") + ("\n".join(blocks))

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
