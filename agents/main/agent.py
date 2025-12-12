import time
from pydantic import BaseModel
import aiosqlite
from datetime import datetime, timezone, timedelta

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from agents.main.config import instructions as Instructions
from agents.main.config import servers as Servers
from mcp_agent.human_input.console_handler import console_input_callback

app = MCPApp(name="main-agent", human_input_callback=console_input_callback)

class MainResponse(BaseModel):
    thought_process: list[str] 
    changes: list[str]

async def promptAgent(name, instruction, server_names, prompt):
    start = time.time()
    async with app.run() as agent_app:
        logger = agent_app.logger

        agent = Agent(
            name=name,
            instruction=instruction,
            server_names=server_names,
        )

        convertor_agent = Agent(
            name="convertor-agent",
            instruction=Instructions.convertor,
            server_names=["think-mcp","sequential-thinking"]
        )

        async with agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            convertor_llm = await convertor_agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate_str(
                message=prompt,
                request_params=RequestParams(
                    max_iterations=30  # Set your desired limit
                ),
            )
            converted_result = await convertor_llm.generate_structured(
                message=result,
                response_model=MainResponse,
            )
            
            end = time.time()
            logger.info(f"[{name}] Worked for {end-start}")
            return converted_result

async def uploadChanges(text):
    async with aiosqlite.connect("files.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)

        async with db.execute("SELECT COUNT(*) FROM changes") as cur:
            (count,) = await cur.fetchone()
        if count == 0:
            await db.execute("DELETE FROM sqlite_sequence WHERE name='changes'")
            await db.commit()

        await db.execute("""
            INSERT INTO changes (content, created_at)
            VALUES (?, ?)
        """, (text, datetime.now(timezone(timedelta(hours=7)))))
        await db.commit()

async def central(prompt):
    response = await promptAgent("centralized_agent", Instructions.centralized, Servers.centralized, prompt)
    formattedResponse = f"Thought process:\n{'\n'.join(response.thought_process) if response.thought_process else '-'}\n\nChanges:\n{'\n'.join(response.changes) if response.changes else '-'}"
    await uploadChanges(formattedResponse)