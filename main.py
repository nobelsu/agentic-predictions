import time
import asyncio
import csv
import subprocess
import aiosqlite

import agents.main.agent as MainAgent
from agents.prediction.helper import row_to_formatted_string

async def getLatestReport():
    async with aiosqlite.connect("files.db") as db:
        async with db.execute("""
            SELECT content FROM reports
        ORDER BY created_at DESC
        LIMIT 1
        """) as cursor:
            row = await cursor.fetchone()

    return row[0] if row else None

async def train():
    start = time.time()
    with open("data/test.csv", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = []
        actual = []
        for row in reader:
            rows.append(row_to_formatted_string(row))
            actual.append(row.get("success"))
            if len(rows) >= 10:
                proc_res = subprocess.run(["uv", "run", "agents/prediction/agent.py", "--p"] + rows + ["--a"] + actual, capture_output=True, text=True)
                instructions = await getLatestReport()

                stdOut = await MainAgent.summarize(proc_res.stdout)
                if not stdOut:
                    stdOut = ""
                stdErr = await MainAgent.summarize(proc_res.stderr)
                if not stdErr:
                    stdErr = ""
                augmentedInstructions = instructions + "\n\nSTDOUT:\n" + stdOut + "\n\nSTDERR:\n" + stdErr

                print("!!AUGMENTED INSTRUCTIONS START!!\n\n",augmentedInstructions,"\n\n!!AUGMENTED INSTRUCTIONS END!!\n")

                await MainAgent.central(augmentedInstructions)
                rows = []
                actual = []
    end = time.time()
    t = end - start
    print(f"\n\nTotal run time: {t:.2f}s")

async def testPredict():
    start = time.time()
    with open("data/train.csv", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = []
        actual = []
        for row in reader:
            rows.append(row_to_formatted_string(row))
            actual.append(row.get("success"))
            if len(rows) >= 1:
                proc_res = subprocess.run(["uv", "run", "agents/prediction/agent.py", "--p"] + rows + ["--a"] + actual, capture_output=True, text=True)
                instructions = await getLatestReport()
                stdOut = await MainAgent.summarize(proc_res.stdout)
                if not stdOut:
                    stdOut = ""
                stdErr = await MainAgent.summarize(proc_res.stderr)
                if not stdErr:
                    stdErr = ""
                augmentedInstructions = instructions + "\n\nSTDOUT:\n" + stdOut + "\n\nSTDERR:\n" + stdErr

                print("!!AUGMENTED INSTRUCTIONS START!!\n\n",augmentedInstructions,"\n\n!!AUGMENTED INSTRUCTIONS END!!\n")

                await MainAgent.central(augmentedInstructions)
                rows = []
                actual = []
                break
    end = time.time()
    t = end - start
    print(f"\n\nTotal run time: {t:.2f}s")

if __name__ == "__main__":
    asyncio.run(train())