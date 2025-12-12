import time
from pydantic import BaseModel
import aiosqlite
from datetime import datetime, timezone, timedelta

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from agents.main.config import instructions as Instructions
from agents.main.config import servers as Servers
from mcp_agent.human_input.console_handler import console_input_callback
from mcp_agent.app import MCPApp
from mcp_agent.workflows.factory import AgentSpec, OrchestratorOverrides, create_orchestrator

app = MCPApp(name="main-agent", human_input_callback=console_input_callback)

async def promptOrchestrator(prompt):
    async with app.run() as running_app:
        orchestrator = create_orchestrator(
            available_agents=[
                AgentSpec(
                    name="researcher",
                    instruction="""
                    You are the Researcher Worker Agent.

                    Your job is to gather all external information required to improve the agent, using the allowed tools.

                    Use sequential thinking to reason about this.

                    Your responsibilities:
                    1. Use mcp-registry and mcp-compass to find MCP servers
                    2. Use context7 and github to find documentation for MCP servers or other code
                    3. Use crawl4ai for web crawling more generally. Some MCP directories to ground your search on include:
                    - https://registry.modelcontextprotocol.io/
                    - https://mcpservers.org/
                    - https://www.pulsemcp.com/servers
                    - https://mcp.so/
                    - https://mcpserverhub.com/
                    4. Produce a structured, ranked list of candidate MCP servers that do NOT require API keys, including:
                    - name
                    - expected usefulness
                    - source link(s)
                    - any relevant implementation details
                    5. Identify opportunities for:
                    - increasing F_0.5 score, precision, recall and accuracy
                    - improving signal extraction
                    - augmenting reasoning capability via free MCP servers
                    6. Analyze environment issues such as:
                    - logs
                    - errors
                    - misconfigurations

                    Present results clearly and logically for the Instructor to approve.

                    You never modify code. You gather data and analyze.
                    """,
                    server_names=["sequential-thinking","crawl4ai", "mcp-compass", "mcp-registry","context7","github"],
                ),
                AgentSpec(
                    name="programmer",
                    instruction="""
                    You are the Programmer Worker Agent.

                    Your job is to apply minimal, precise, and safe code changes requested by the Instructor to improve the prediction agent.

                    Use sequential-thinking to reason about this.

                    Your responsibilities:
                    1. Operate strictly within: ~/Desktop/Vela/vela-temp/agents/prediction
                    2. Implement only the changes explicitly approved by the Planner.
                    3. Apply targeted patches, including:
                    - Adding selected MCP servers
                    - Fixing bugs found in logs
                    - Improving prediction logic when allowed
                    4. Never modify the PredictionResponse class or alter output format.
                    5. Do NOT provide or load CSV data into the agent.
                    6. When editing files, always return:
                    - A unified diff patch
                    - A clear explanation of what changed and why
                    7. Avoid refactoring unrelated components.
                    8. Ensure all changes have minimal footprint and are reversible.
                    9. Use filesystem to view, then edit files
                    10. Use context7 and github to find documentation for MCP servers or other code 
                    11. Use crawl4ai for web crawling and even deeper searchers of documentation and code examples

                    You are the implementer.  
                    """,
                    server_names=["context7","github","filesystem","crawl4ai","sequential-thinking"],
                ),
                AgentSpec(
                    name="reasoner",
                    instruction="""
                    You are the Reasoner Agent.

                    Your job is to provide deep, structured, sequential reasoning that guides the Instructor's decisions.

                    Your responsibilities:
                    1. Think step-by-step and reason explicitly. Use sequential-thinking to reason about this.
                    2. Analyze:
                    - Which MCP servers will provide the most value
                    - Which changes are safe vs. risky
                    - Why model performance is limited
                    - Where logs indicate bugs or failures
                    3. Evaluate:
                    - trade-offs
                    - potential side effects
                    - predicted impact on F_0.5, precision, recall, and accuracy
                    4. Provide reasoning structures such as:
                    - decision trees
                    - ranked recommendations
                    - risk assessments
                    - detailed error analyses

                    You do not write code.

                    You may use filesystem ONLY to view files, not to edit them.

                    You generate high-quality reasoning for the Instructor.
                    """,
                    server_names=["sequential-thinking","think-mcp","filesystem"]
                ),
            ],
            plan_type="iterative", 
            overrides=OrchestratorOverrides(
                planner_instruction="""
                    You are the Planner.

                    Your job is to coordinate and supervise a Programmer, a Researcher, and a Reasoner working together to improve a prediction AI agent's performance.

                    For context:

                    You do not write code or perform research. You assign tasks, enforce constraints, and ensure correctness.
                    
                    Your responsibilities:
                    1. Define sequential tasks for the Programmer, Researcher, and Thinker/Reasoner.
                    2. Enforce all constraints:
                    - Make only minimal, necessary code changes.
                    - No unrelated changes.
                    - The agent must NOT gain access to the CSV data.
                    - MCP servers added must not require API keys.
                    3. Assign tasks such as:
                    - Identifying MCP servers that are API-key-free
                    - Finding relevant documentation
                    - Diagnosing and fixing issues in STDOUT and STDERR logs
                    - Implementing features into the agent's code
                    - Planning improvements to F_0.5, recall, precision and accuracy
                    - Reasoning about the performance of the prediction agent 
                    - Thinking about the prediction agent's weaknesses
                    
                    Evaluate results and require revisions when needed.
                    Ensure all improvements increase model quality safely.

                    You define the plan. Workers execute it.
                """,
                synthesizer_instruction="""
                Your output should be a report on the following: 
                - The approved MCP servers being added
                - The final code patches
                - The rationale summary
                - Remaining risks or follow-up actions
                """,
            ),
            provider="google",
            context=running_app.context,
        )

        return await orchestrator.generate_str(prompt, request_params=RequestParams(max_iterations=30))

async def promptAgent(name, instruction, server_names, prompt, iter, model):
    start = time.time()
    async with app.run() as agent_app:
        logger = agent_app.logger

        agent = Agent(
            name=name,
            instruction=instruction,
            server_names=server_names,
        )

        # convertor_agent = Agent(
        #     name="convertor-agent",
        #     instruction=Instructions.convertor,
        #     server_names=["think-mcp","sequential-thinking"]
        # )

        async with agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            # convertor_llm = await convertor_agent.attach_llm(GoogleAugmentedLLM)
            result = await llm.generate_str(
                message=prompt,
                request_params=RequestParams(
                    max_iterations=iter,
                    model=model  # Set your desired limit
                ),
            )
            # converted_result = await convertor_llm.generate_structured(
            #     message=result,
            #     response_model=MainResponse,
            #     request_params=RequestParams(
            #         model="gemini-2.5-flash"  # Set your desired limit
            #     ),
            # )
            
            end = time.time()
            logger.info(f"[{name}] Worked for {end-start}")
            return result

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
    # response = await promptOrchestrator(prompt)
    response = await promptAgent("centralized_agent", Instructions.centralized, Servers.centralized, prompt, 20, "gemini-3-pro-preview")
    # formattedResponse = f"Thought process:\n{'\n'.join(response.thought_process) if response.thought_process else '-'}\n\nChanges:\n{'\n'.join(response.changes) if response.changes else '-'}"
    await uploadChanges(response)

async def summarize(prompt):
    response = await promptAgent("summarizer_agent", Instructions.summarizer, [], prompt, 10, "gemini-2.5-flash")
    return response