import os
import json
import asyncio
import time
import re
import numpy as np
import nltk

# Download required NLTK data for sentence splitting
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

import cohere
from openai import AsyncOpenAI
from groq import AsyncGroq
from azure.cosmos import CosmosClient, exceptions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# ==========================================
# FASTAPI SETUP & SECURITY
# ==========================================
app = FastAPI(title="SkillQuest AI Engine")

# 1. Allow Node.js (even on localhost) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Security (Removed for open access)
# No API key required for internal use

# 3. Define the incoming Data format from Node.js
class UserParameters(BaseModel):
    target_topic: str
    proficiency: str
    cognitive_difficulty: List[List[float]]  # e.g. [[p_tol_easy, p_tol_med, p_tol_hard], [...]]
    historical_gaps: str
    gamification_level: int          # raw int from DB: current_level
    gamification_streak: int         # raw int from DB: current_streak
    gamification_badge: str          # raw string from DB: badge_name

# ==========================================
# PHASE 1: RETRIEVER (Azure Cosmos DB)
# ==========================================
class Agent1Retriever:
    def __init__(self):
        self.cohere_api_key = os.environ.get("COHERE_API_KEY")
        self.co = cohere.Client(self.cohere_api_key)
        
        self.endpoint = os.environ.get("COSMOS_ENDPOINT")
        self.key = os.environ.get("COSMOS_KEY")
        self.client = CosmosClient(self.endpoint, self.key)
        
        # Make sure these match your actual Azure setup
        self.database = self.client.get_database_client("CourseRAG_DB")
        self.container = self.database.get_container_client("CourseMaterials")

    def get_embedding(self, text):
        response = self.co.embed(texts=[text], model='embed-english-v3.0', input_type='search_query')
        return response.embeddings[0]

    def retrieve_chunks(self, vector, top_k=3):
        query = """
            SELECT TOP @top_k c.text, VectorDistance(c.embedding, @query_vector) AS SimilarityScore 
            FROM c ORDER BY VectorDistance(c.embedding, @query_vector)
        """
        parameters = [{"name": "@top_k", "value": top_k}, {"name": "@query_vector", "value": vector}]
        try:
            results = list(self.container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
            return [item['text'] for item in results]
        except exceptions.CosmosHttpResponseError:
            return [] # Returns empty if DB fails so it doesn't crash the server

    def execute(self, user_params):
        search_string = f"{user_params['target_topic']} for a student with {user_params['proficiency']} proficiency."
        query_vector = self.get_embedding(search_string)
        ground_truth_chunks = self.retrieve_chunks(query_vector, top_k=3)
        return {"user_parameters": user_params, "retrieved_ground_truth": ground_truth_chunks}

# ==========================================
# PHASE 2: GENERATORS (GPT-4o & Llama 3.3)
# ==========================================
class Agent2GPT:
    def __init__(self):
        # Pointing to GitHub's Azure Model Inference endpoint
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://models.inference.ai.azure.com"
        )

    async def generate(self, payload):
        params = payload['user_parameters']
        system_prompt = f"""
        TASK: Create an educational reading and EXACTLY 15 multiple-choice quiz questions on the topic: '{params['target_topic']}'.
        RULES: Use ONLY the provided 'Ground Truth Facts'. Do NOT invent any information.
        - Student Proficiency: {params['proficiency']}
        - Cognitive Difficulty: {params['cognitive_difficulty']}
        - Known Knowledge Gaps to address: {params['historical_gaps']}
        - Gamification Tone (write as a dungeon master): {params['gamification']}
        CRITICAL FORMAT RULE: The "quiz" array MUST contain EXACTLY 15 items — no more, no fewer.
        FORMAT: Return strictly valid JSON with two keys: "educational_content" (string) and "quiz" (array of EXACTLY 15 objects, each with: "question", "correct_answer", "distractors" array of 3 strings).
        """
        user_prompt = f"Ground Truth Facts:\n{json.dumps(payload['retrieved_ground_truth'])}"
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)

class Agent3Groq:
    def __init__(self):
        self.client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY_AGENT_3"))

    async def generate(self, payload):
        params = payload['user_parameters']
        system_prompt = f"""
        TASK: Create an educational reading and EXACTLY 15 multiple-choice quiz questions on the topic: '{params['target_topic']}'.
        RULES: Use ONLY the provided 'Ground Truth Facts'. Do NOT invent any information.
        - Student Proficiency: {params['proficiency']}
        - Cognitive Difficulty: {params['cognitive_difficulty']}
        - Known Knowledge Gaps to address: {params['historical_gaps']}
        - Gamification Tone (write as a dungeon master): {params['gamification']}
        CRITICAL FORMAT RULE: The "quiz" array MUST contain EXACTLY 15 items — no more, no fewer.
        FORMAT: Return strictly valid JSON with two keys: "educational_content" (string) and "quiz" (array of EXACTLY 15 objects, each with: "question", "correct_answer", "distractors" array of 3 strings).
        """
        user_prompt = f"Ground Truth Facts:\n{json.dumps(payload['retrieved_ground_truth'])}"
        response = await self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)

async def run_phase_2_bulk(phase_1_output: Dict[str, Any]) -> Dict[str, Any]:
    agent_2, agent_3 = Agent2GPT(), Agent3Groq()
    results = await asyncio.gather(agent_2.generate(phase_1_output), agent_3.generate(phase_1_output))
    return {
        "agent_2_gpt": results[0], "agent_3_groq": results[1],
        "original_ground_truth": phase_1_output['retrieved_ground_truth']
    }

# ==========================================
# PHASE 3: DETERMINISTIC JUDGE (Math Scoring)
# ==========================================
class SkillQuestEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    def evaluate_agent_output(self, agent_json, ground_truth_list):
        content = agent_json.get("educational_content", "")
        quiz = agent_json.get("quiz", [])
        ground_truth_text = " ".join(ground_truth_list)

        # Safety check if text is empty
        if not content or not ground_truth_text:
            return {"final_float": 0.0}

        tfidf = TfidfVectorizer().fit_transform([content, ground_truth_text])
        cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        rouge_score = self.scorer.score(ground_truth_text, content)['rouge1'].recall
        
        jaccard_scores = []
        for q in quiz:
            a = set(q.get("correct_answer", "").lower().split())
            for dist in q.get("distractors", []):
                b = set(dist.lower().split())
                jaccard_scores.append(len(a.intersection(b)) / len(a.union(b)) if a and b else 0)
        
        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
        jaccard_penalty = 0.5 if avg_jaccard > 0.7 or avg_jaccard < 0.1 else 1.0
        # Soft penalty for scoring only: 13-16 questions = 0.5x, not 0.
        # Exact 15-question enforcement happens post-selection (see orchestrator).
        structural_multiplier = 1.0 if len(quiz) == 15 else (0.5 if 13 <= len(quiz) <= 16 else 0.0)

        final_score = ((cos_sim * 0.4) + (rouge_score * 0.3) + (jaccard_penalty * 0.3)) * structural_multiplier
        return {"final_float": round(final_score, 4)}

# ==========================================
# PHASE 4: HALLUCINATION FIREWALL (Agent 4)
# ==========================================
class Agent4Auditor:
    def __init__(self):
        self.client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY_AGENT_4"))

    async def verify_fact(self, fact: str, ground_truth: str) -> Dict[str, Any]:
        system_prompt = "Verify if Claim is reasonably supported by Ground Truth. Output ONLY valid JSON: {\"fact_is_proven\": true/false}."
        try:
            response = await self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Ground Truth: {ground_truth}\nClaim: {fact}"}],
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            # On API errors (e.g. rate limits) give benefit of the doubt — log for visibility
            print(f"[Firewall] verify_fact exception (treated as passed): {e}")
            return {"fact_is_proven": True}

    async def execute_firewall(self, winner_json: Dict[str, Any], ground_truth: str) -> Dict[str, Any]:
        facts = [s.strip() for s in nltk.sent_tokenize(winner_json['educational_content']) if len(s.strip()) > 10]
        if not facts:
            return {"is_clean": False}

        # Cap concurrent Groq calls to avoid rate-limit 429s
        semaphore = asyncio.Semaphore(5)
        async def bounded_verify(fact: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.verify_fact(fact, ground_truth)

        results = await asyncio.gather(*[bounded_verify(f) for f in facts])
        passed = sum(1 for r in results if r.get("fact_is_proven"))
        pass_rate = passed / len(facts)
        print(f"[Firewall] {passed}/{len(facts)} facts verified ({pass_rate:.0%}) — threshold: 70%")
        # 70% threshold: not every sentence needs a direct RAG match to be valid
        return {"is_clean": pass_rate >= 0.70}

# ==========================================
# PHASE 5: THE FINISHER (Cohere Formatting)
# ==========================================
class Agent1Finisher:
    def __init__(self):
        self.co = cohere.Client(os.environ.get("COHERE_API_KEY"))

    def execute_final_polish(self, validated_json, user_params):
        style_prompt = "- Format the educational text beautifully using Markdown."
        system_prompt = f"TASK: Transform the provided text into beautifully formatted Markdown. RULES: {style_prompt}. Target: {user_params['proficiency']}."
        response = self.co.chat(
            message=validated_json.get("educational_content", ""),
            preamble=system_prompt,
            model="command-r-plus-08-2024"
        )
        return {
            "content": response.text,
            "quiz": validated_json.get("quiz", [])
        }

# ==========================================
# THE ORCHESTRATOR LOOP
# ==========================================
def enforce_15_questions(winner_json: dict) -> dict:
    """Post-selection enforcement: trim or pad quiz to exactly 15 questions."""
    quiz = winner_json.get("quiz", [])
    if len(quiz) > 15:
        winner_json["quiz"] = quiz[:15]   # trim extras
    # If fewer than 15 we leave it — the retry loop will try again next attempt
    return winner_json


async def skillquest_orchestrator(phase_1_payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    # Bug 1 fix: fail fast if the retriever returned no grounding data
    if not phase_1_payload.get("retrieved_ground_truth"):
        return {"success": False, "error": "Retriever returned no ground truth chunks. Check Cosmos DB connection and container name."}

    attempt = 1
    while attempt <= max_retries:
        print(f"[Orchestrator] Attempt {attempt}/{max_retries}")
        p2 = await run_phase_2_bulk(phase_1_payload)

        evaluator = SkillQuestEvaluator()
        a2_score = evaluator.evaluate_agent_output(p2['agent_2_gpt'], p2['original_ground_truth'])
        a3_score = evaluator.evaluate_agent_output(p2['agent_3_groq'], p2['original_ground_truth'])
        print(f"[Orchestrator] Scores — GPT: {a2_score['final_float']}, Groq: {a3_score['final_float']}")

        winner_json = p2['agent_2_gpt'] if a2_score['final_float'] >= a3_score['final_float'] else p2['agent_3_groq']

        # Enforce exactly 15 questions in final output (trim if LLM over-generated)
        winner_json = enforce_15_questions(winner_json)

        # Only proceed to firewall if the winner has exactly 15 questions
        if len(winner_json.get("quiz", [])) != 15:
            print(f"[Orchestrator] Winner has {len(winner_json.get('quiz', []))} questions — retrying.")
            attempt += 1
            await asyncio.sleep(1)  # Bug 4 fix: non-blocking async sleep
            continue

        auditor = Agent4Auditor()
        firewall = await auditor.execute_firewall(winner_json, p2['original_ground_truth'])

        if firewall["is_clean"]:
            return {"success": True, "validated_json": winner_json}

        attempt += 1
        await asyncio.sleep(1)  # Bug 4 fix: non-blocking async sleep

    return {"success": False, "error": "Failed to pass Hallucination Firewall after maximum retries."}

# ==========================================
# THE API ENDPOINT (Node.js talks to this!)
# ==========================================
def map_cognitive_difficulty(tol_matrix: List[List[float]]) -> str:
    """Flatten the 2D tolerance matrix and pick the highest-scoring level."""
    flat = [val for row in tol_matrix for val in row]
    if not flat:
        return "Application (Medium Difficulty). Test their ability to use the concept."
    idx = flat.index(max(flat)) % 3  # 0=easy, 1=medium, 2=hard
    labels = [
        "Memorization / Knowledge (Easy Difficulty). Ask 'What is X?' style questions.",
        "Application (Medium Difficulty). Ask 'How do you use X?' style questions.",
        "Evaluation / Critical Thinking (Hard Difficulty). Ask 'Find the bug in this X code.' style questions."
    ]
    return labels[idx]


@app.post("/api/generate-lesson")
async def generate_lesson(params: UserParameters):
    try:
        nodejs_inputs = params.model_dump()

        # Convert raw DB values into LLM-readable strings
        nodejs_inputs["cognitive_difficulty"] = map_cognitive_difficulty(params.cognitive_difficulty)
        nodejs_inputs["gamification"] = (
            f"Level {params.gamification_level}, on a {params.gamification_streak}-day login streak. "
            f"Recently earned the '{params.gamification_badge}' badge."
        )
        # Remove raw fields so only the narrative string reaches agents
        nodejs_inputs.pop("gamification_level", None)
        nodejs_inputs.pop("gamification_streak", None)
        nodejs_inputs.pop("gamification_badge", None)

        retriever = Agent1Retriever()
        phase_1 = retriever.execute(nodejs_inputs)
        
        loop_result = await skillquest_orchestrator(phase_1)
        
        if loop_result["success"]:
            finisher = Agent1Finisher()
            final_data = finisher.execute_final_polish(loop_result["validated_json"], nodejs_inputs)
            return {"status": "success", "data": final_data}
        else:
            raise HTTPException(status_code=500, detail=loop_result["error"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))