from pydantic import BaseModel
import asyncio
import argparse
import traceback
from datetime import datetime, timezone, timedelta
import aiosqlite
import csv
import time
import json

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
        level="info",
        progress_display=True,
        path_settings={
            "path_pattern":"logs/prediction/prediction-{unique_id}.jsonl",
            "unique_id": "timestamp",
            "timestamp_format": "%Y%m%d_%H%M%S"
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
            "web-search-mcp": MCPServerSettings( 
                command="npx",
                args=["-y", "@guhcostan/web-search-mcp@latest"]
            )
        }
    ),
    google=GoogleSettings(default_model="gemini-2.5-flash"),
)

async def predictSuccess(prompts, success_values):
    app = MCPApp(settings=settings)

    async with app.run() as agent_app:
        prediction_instruction = f"""
            You are a skeptical, precision-focused Venture Capital Analyst. Your task is to predict if a startup will be an "Outlier Success" (Exit > $500M or Funding > $500M).

            **The Base Rate is < 0.1%. Default to FALSE.**

            **Input Data (Anonymized):**
            *   `industry`: Startup's sector.
            *   `ipos` / `acquisitions`: Count of *verified* exits.
            *   `educations_json`: Degree, Institution Rank (QS).
            *   `jobs_json`: Role, Company Size, Industry, Duration.
            *   `anonymised_prose`: Narrative summary.

            **Crucial Constraint**: The data is anonymized. You CANNOT search for specific company names. You MUST rely on the *metadata* (Company Size, Rank, Role).

            **Your Workflow:**
            1.  **Analyze Metadata Signals**:
                *   **"Unicorn" Executive (Tier 1)**: 
                    *   **C-Level** (CEO, CTO, CPO) at a **Product Tech** Company with Size > 2000.
                    *   **Founder/CEO** at a **Fintech/Financial Services** Company with Size > 500.
                *   **"Unicorn" Executive (Tier 2)**:
                    *   **VP/SVP** at a **Product Tech** Company with Size > 10000 (Global Giant).
                    *   **Global Head** of a *Core* Technical Function (Security, AI, Cloud, Engineering, Data) at a Global Giant (Size > 10000).
                    *   **Head of Strategy** at a Global Giant (Size > 10000).
                *   **"Unicorn" Product/Growth Leader (Tier 3)**:
                    *   **Director** or **Senior Director** at a Global Giant (Size > 10000).
                    *   **Product Line Manager**, **Group Product Manager**, **Head of Design** at a Global Giant (Size > 10000).
                    *   **General Manager (GM)** or **Head of Expansion** at a Tech/Fintech Company > 1000.
                    *   **Production Director** or **Executive Producer** at a **Gaming** Company > 1000.
                *   **Senior Technical Lead**:
                    *   **Principal Engineer**, **Distinguished Engineer**, **Fellow**, **Chief Architect** at a Tech Company > 5000.
                    *   *EXCLUDE*: Any title with "Lead" (e.g. Lead Engineer, Dev Lead), "Senior", "Staff". "Principal" is the minimum bar.
                *   **Proven Founder**: 
                    *   `ipos` > 0 OR `acquisitions` > 0.
                    *   **CRITICAL CHECK**: If `acquisitions` > 0, check the prose. If it says "Undisclosed", TREAT AS ZERO (0). Only count if it implies a major exit.
                *   **Elite Technical**: PhD/Masters in CS/Engineering/Physics/Math from a Top 50 QS Ranked University.
                *   **Elite Business**: MBA from a Top 10 QS Ranked University.
                *   **Deep Domain Expert**: 
                    *   PhD/MD/Professor in **Life Sciences, Biotechnology, Pharmaceuticals, Semiconductors, Physics, Mathematics**.
                    *   *EXCLUDE*: Practitioners (MD/DDS) without research roles.

            2.  **Contextual Search (General)**:
                *   Use `web-search-mcp` to check the *base rate* for the specific *Role + Industry* combination.
                *   Query: "Success rate of [Industry] startups founded by former [Job Title]".
                *   *Do NOT search for the specific person or company name.*

            3.  **Evaluate & Weight**:
                *   **AUTO-PASS (+3)**: `ipos` > 0 OR (`acquisitions` > 0 AND NOT Undisclosed).
                *   **AUTO-PASS (+3)**: "Unicorn Executive (Tier 1)" (C-Level at Large Product Tech OR Fintech Founder > 500).
                *   **STRONG POSITIVE (+2)**: "Unicorn Executive (Tier 2)" (VP at Global Giant OR Core Tech Head).
                *   **POSITIVE (+1.5)**: Deep Domain Expert (PhD/Professor/Lab Chief) in **Hard Sciences/Tech**. (Do not double count with "Research Scientist").
                *   **POSITIVE (+1.5)**: "Unicorn Product/Growth Leader (Tier 3)" (PLM/GPM/GM/Producer at Global Giant/Scaleup).
                *   **POSITIVE (+1.5)**: Senior Technical Lead (Principal/Fellow) at Tech Co > 5000.
                *   **WEAK POSITIVE (+1)**: Elite Technical OR Elite Business Education. (**MAX +1** total for education).
                *   **WEAK POSITIVE (+1)**: "Long-Tenure Founder" (>5 years) in "Real World" industries (Logistics, Manufacturing) - *EXCLUDE* Real Estate/Construction/Consulting/Healthcare/Software/Internet.
                *   **WEAK POSITIVE (+1)**: Creative Director / Design Lead (only for **Consumer/Media** startups).
                *   **WEAK POSITIVE (+1)**: "Grant Winner" (SBIR, STTR, ARPA-E) or "Research Scientist" (implies non-dilutive funding/deep tech).
                *   **WEAK POSITIVE (+0.5)**: MD/DDS (Practitioner) without PhD.
                *   **WEAK POSITIVE (+0.5)**: Software Engineer / Data Scientist at Global Giant (10000+).
                *   **NEGATIVE (-2)**: "Founder" of a small company (Size < 50) with NO verified large exits AND NO other positive signals.
                *   **NEGATIVE (-1)**: Non-product role (Sales, Marketing, HR, Legal, BizDev) as the *highest* role achieved.

            4.  **Reason via Sequential Thinking**:
                *   Step 1: Extract all "Positive" and "Negative" points based on the rules above.
                *   Step 2: STRICTLY apply the "Unicorn Executive" definitions. **DISQUALIFY IT SERVICES/CONSULTING.** **DISQUALIFY SALES/BIZDEV.**
                *   Step 3: Check for "Deep Domain Expert".
                *   Step 4: Sum the evidence. Does the profile look like the top 0.01%?

            5.  **Final Prediction**: 
                *   If (Score >= +2.5), predict **TRUE**.
                *   If (Score >= +2.0 AND has at least 2 distinct positive signals, one of which is >= +1.5), predict **TRUE**.
                *   Otherwise, predict **FALSE**.

            **Output Format**:
            1. prediction: True/False
            2. reason: A concise paragraph citing the specific signals (e.g., "Previous acquisition count: 1 (Verified)", "VP at 10k+ employee tech firm", "Principal Engineer at Major Semiconductor Co").

            DO NOT output anything else.
        """

        agent = Agent(
            name="prediction-agent",
            instruction=prediction_instruction,
            server_names=["crawl4ai","sequential-thinking","web-search-mcp"],
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
                        if not prompt or len(prompt.strip()) == 0:
                             results.append(PredictionResponse(
                                prediction=False,
                                reason="No input data provided.",
                            ))
                             continue

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

                with open("values.csv", "a", newline="", encoding="utf-8") as f:
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

async def default():
    parser = argparse.ArgumentParser(description="Prediction Agent script")
    parser.add_argument("--p",nargs="+",help="List of prompts to provide agent")
    parser.add_argument("--s",nargs="+",help="Actual success values")

    args = parser.parse_args()
    prompts = args.p
    success_values = args.s
    
    report = await predictSuccess(prompts, success_values)
    await upload(report)

def formatRow(row):
    def parse_json_field(field):
        if not field or field.strip() == "":
            return []
        try:
            return json.loads(field)
        except Exception:
            return [] 
    
    industry = row.get("industry", "")
    ipos = parse_json_field(row.get("ipos"))
    acquisitions = parse_json_field(row.get("acquisitions"))
    educations_json = parse_json_field(row.get("educations_json"))
    jobs_json = parse_json_field(row.get("jobs_json"))
    anonymised_prose = row.get("anonymised_prose", "").strip()

    formatted = f"""
        industry: "{industry}",
        ipos: {json.dumps(ipos)},
        acquisitions: {json.dumps(acquisitions)},
        educations_json: {json.dumps(educations_json)},
        jobs_json: {json.dumps(jobs_json)},
        anonymised_prose: \"\"\"
            {anonymised_prose}
        \"\"\"
        """

    return formatted

if __name__ == "__main__":
    start = time.time()
    with open("data/test.csv", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        prompts = []
        success_values = []
        for row in reader:
            prompts.append(formatRow(row))
            success_values.append(row.get("success"))
    print("[TESTING] Running predictions on test file...")
    report = asyncio.run(predictSuccess(prompts, success_values))
    print("[TESTING] Finished running predictions on test file!")
    print("[TESTING] Uploading report...")
    asyncio.run(upload(report))
    print("[TESTING] Uploaded report!")
    end = time.time()
    t = end - start
    print(f"[TESTING] Total run time: {t:.2f}s")

# DO NOT CHANGE END
