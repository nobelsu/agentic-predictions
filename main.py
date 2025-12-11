import time
import asyncio
import csv
import subprocess
import sqlite3
from datetime import datetime, timezone, timedelta
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

async def uploadChanges(text):
    async with aiosqlite.connect("files.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS changes (
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
            INSERT INTO changes (content, created_at)
            VALUES (?, ?)
        """, (text, datetime.now(timezone(timedelta(hours=7)))))
        await db.commit()

async def train():
    start = time.time()
    with open("data/train.csv", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = []
        actual = []
        for row in reader:
            rows.append(row_to_formatted_string(row))
            actual.append(row.get("success"))
            if len(rows) >= 10:
                subprocess.run(["uv", "run", "agents/prediction/agent.py", "--p"] + rows + ["--a"] + actual)
                instructions = await getLatestReport()
                changes = await MainAgent.central(instructions)
                await uploadChanges(changes)
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
                subprocess.run(["uv", "run", "agents/prediction/agent.py", "--p"] + rows + ["--a"] + actual)
                instructions = await getLatestReport()
                print(instructions)
                rows = []
                actual = []
                break
    end = time.time()
    t = end - start
    print(f"\n\nTotal run time: {t:.2f}s")

if __name__ == "__main__":
    asyncio.run(train())