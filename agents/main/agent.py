import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from agents.main.config import instructions as Instructions
from agents.main.config import servers as Servers
from mcp_agent.human_input.console_handler import console_input_callback

app = MCPApp(name="main-agent", human_input_callback=console_input_callback)

async def promptAgent(name, instruction, server_names, prompt):
    start = time.time()
    async with app.run() as agent_app:
        logger = agent_app.logger

        agent = Agent(
            name=name,
            instruction=instruction,
            server_names=server_names,
        )

        async with agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            result = await llm.generate_str(
                message=prompt,
                request_params=RequestParams(
                    max_iterations=30  # Set your desired limit
                )
            )
            
            end = time.time()
            logger.info(f"[{name}] Worked for {end-start}")
            return result

async def central(prompt):
    return await promptAgent("centralized_agent", Instructions.centralized, Servers.centralized, prompt)