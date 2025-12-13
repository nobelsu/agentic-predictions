import time
import csv
import subprocess

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.human_input.console_handler import console_input_callback

from utils.csv import formatRow
from utils.files import writeText
from utils.sqlite import getLatest, upload 

from agents.main.util_agents import summarize, extract
import agents.main.config as Config

app = MCPApp(name="improver-agent", human_input_callback=console_input_callback)

async def improve(prompt):
    async with app.run():

        agent = Agent(
            name="improver-agent",
            instruction=Config.instruct_improve,
            server_names=Config.server_improve,
        )

        async with agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)

            changes = await llm.generate(
                message=prompt,
                request_params=RequestParams(
                    max_iterations=Config.iter_improve,
                    model=Config.model_improve
                ),
            ) 

            extracted_changes = await extract(str(changes))

            return extracted_changes

async def train(row_start=1, row_end=100000, csv_path="data/train.csv", step_size=10):
    async with app.run() as agent_app:

        print("[TRAINING] Starting training...")

        start = time.time()

        with open(csv_path, newline="", encoding="utf-8") as f_in:

            reader = csv.DictReader(f_in)
            rows = []
            success_values = []
            row_counter = 0

            agent = Agent(
                name="improver-agent",
                instruction=Config.instruct_improve,
                server_names=Config.server_improve,
            )

            async with agent:

                print("[TRAINING] Agent initialized...")

                llm = await agent.attach_llm(GoogleAugmentedLLM)

                print("[TRAINING] LLM attached...")

                for row in reader:

                    row_counter += 1

                    if row_counter < row_start:
                        continue
                    if row_counter > row_end:
                        break

                    rows.append(formatRow(row))
                    success_values.append(row.get("success"))

                    if len(rows) >= step_size:
                        
                        print(f"[TRAINING] Running predictions on rows {row_counter-step_size+1} to {row_counter}...")

                        proc_res = subprocess.run(["uv", "run", "agents/prediction/prediction_agent.py", "--p"] + rows + ["--s"] + success_values, capture_output=True, text=True)
                        
                        print(f"[TRAINING] Finished predictions on rows {row_counter-step_size+1} to {row_counter}!")

                        writeText("utils/text/std_out.txt", proc_res.stdout)
                        writeText("utils/text/std_err.txt", proc_res.stderr)

                        instructions = await getLatest("reports", "files.db")
                        stdOut = await summarize("utils/text/std_out.txt")
                        stdErr = await summarize("utils/text/std_err.txt")

                        if not stdOut:
                            stdOut = ""
                        if not stdErr:
                            stdErr = ""
                        
                        augmentedInstructions = instructions + "\n\nSTDOUT:\n" + stdOut + "\n\nSTDERR:\n" + stdErr
                        await upload(augmentedInstructions, "std_reports", "files.db")

                        print(f"[TRAINING] Making changes based on rows {row_counter-step_size+1} to {row_counter}...")

                        changes = await llm.generate(
                            message=augmentedInstructions,
                            request_params=RequestParams(
                                max_iterations=Config.iter_improve,
                                model=Config.model_improve
                            ),
                        )

                        print(f"[TRAINING] Finished making changes based on rows {row_counter-step_size+1} to {row_counter}!")
                        
                        extracted_changes = await extract(str(changes))
                        await upload(extracted_changes, "changes", "files.db")
                        
                        rows = []
                        success_values = []
        
        end = time.time()
        t = end - start
        print(f"[TRAINING] Total run time: {t:.2f}s")