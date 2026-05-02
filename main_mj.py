# =========================================================
# FLEXIBLE MULTI-AGENT ORCHESTRATION SYSTEM (13 AGENTS)
# =========================================================

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Tuple
import os, re, json, time
from concurrent.futures import ThreadPoolExecutor

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from tools import search_tool, wiki_tool, save_tool

# ---------------------------------------------------------
# ENV + MODEL
# ---------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.1,
    max_new_tokens=256,
    huggingfacehub_api_token=HF_TOKEN
)
chat_llm = ChatHuggingFace(llm=llm)

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
def call_llm(prompt):
    r = chat_llm.invoke(prompt)
    return r.content if hasattr(r, "content") else str(r)

def safe_json(text):
    try:
        return json.loads(re.search(r"\{.*\}", text, re.DOTALL).group())
    except:
        return {}
def normalize_query(q) -> str:
    """
    Ensure tools always receive a clean single string.
    Fixes: Too many arguments to single-input tool
    """
    if isinstance(q, str):
        return q.strip()

    if isinstance(q, (list, tuple)):
        return " ".join(map(str, q))

    if isinstance(q, dict):
        return " ".join(map(str, q.values()))

    return str(q)
def clean_expansion_entity(text: str) -> str:
    """
    Extract likely entity name from noisy expansion.
    Designed to preserve person names.
    """

    text = normalize_query(text)
    # preserve apostrophes (important for entity names)
    text = re.sub(r"[^a-zA-Z0-9\s']", "", text)
    # remove common question prefixes
    prefixes = [
        "who is", "what is", "tell me about",
        "information about", "details about"
    ]

    lowered = text.lower()
    for p in prefixes:
        if lowered.startswith(p):
            text = text[len(p):].strip()

    # remove trailing descriptive phrases after comma
    #text = re.split(r"[,-]", text)[0].strip()

    # keep first 2–3 capitalized words (likely name)
    words = text.split()

    # preserve full entity phrase if it contains apostrophe
    if "'" in text:
        return text.strip()

    # otherwise fallback to capitalized tokens
    name_tokens = [w for w in words if w and w[0].isupper()]

    if len(name_tokens) >= 2:
        return " ".join(name_tokens[:3])

    # fallback: return cleaned text but not empty
    return text if text else normalize_query(text)
def pretty_print_result(result):
    """
    Smart pretty printer for:
    - QA answers
    - entity dictionaries
    - fallback text
    """

    def line():
        print("=" * 60)

    answer = result.get("answer")
    justification = result.get("justification")
    confidence = result.get("confidence")
    verification = result.get("verification")

    # -------------------------------------------------
    # 🔍 Try to detect entity dictionary inside answer
    # -------------------------------------------------
    parsed_entities = None

    if isinstance(answer, str):
        try:
            parsed_entities = json.loads(answer.replace("'", '"'))
        except Exception:
            try:
                parsed_entities = eval(answer)
            except Exception:
                parsed_entities = None

    # -------------------------------------------------
    # 🟢 CASE 1 — Entity dictionary detected
    # -------------------------------------------------
    if isinstance(parsed_entities, dict):
        line()
        print("📚 DISAMBIGUATED ENTITIES\n")

        for name, info in parsed_entities.items():
            print(f"🔹 {name}")

            if isinstance(info, dict):
                for k, v in info.items():
                    key_clean = k.replace("_", " ").title()
                    print(f"   • {key_clean}: {v}")
            else:
                print(f"   • Info: {info}")

            print()

        print(f"🔢 CONFIDENCE: {confidence}")
        print(f"✅ VERIFIED: {verification}")
        line()
        return

    # -------------------------------------------------
    # 🟢 CASE 2 — Normal QA answer
    # -------------------------------------------------
    line()
    print("📌 ANSWER:\n")
    print(answer)

    if justification:
        print("\n🧾 JUSTIFICATION:\n")
        print(justification)

    print(f"\n🔢 CONFIDENCE: {confidence}")
    print(f"✅ VERIFIED: {verification}")
    line()
# ---------------------------------------------------------
# AGENT 1–3: QUERY UNDERSTANDING
# ---------------------------------------------------------
class QueryProfile(BaseModel):
    intent: str
    complexity: str
    requires_safety: bool
    requires_decomposition: bool
    requires_verification: bool
    is_ambiguous: bool  

class QueryClassifierAgent:

    def detect_query_ambiguity(self, query: str) -> bool:
        """
        Heuristic ambiguity detector.
        Flags underspecified entity queries.
        """
        q = query.lower().strip()
        words = q.split()

        # Pattern: very short person queries
        trigger_words = ["who", "what", "tell", "about"]

        # Case 1: short query + trigger word
        if len(words) <= 3 and any(w in q for w in trigger_words):
            return True

        # Case 2: single-name entity (e.g., "who is anurag")
        if len(words) <= 3 and len(words) >= 2:
            return True

        return False

    def classify(self, query: str) -> QueryProfile:
        q = query.lower()

        # ✅ correct method call
        is_ambiguous = self.detect_query_ambiguity(query)

        return QueryProfile(
            intent=(
                "comparison" if "which" in q or "or" in q else
                "safety" if any(w in q for w in ["safe", "risk", "side effects"]) else
                "factual"
            ),
            complexity="multihop" if q.count("who") + q.count("when") > 1 else "simple",
            requires_safety=any(w in q for w in ["safe", "danger", "toxic"]),
            requires_decomposition=" or " in q or ("who" in q and "when" in q),
            requires_verification="verify" in q or "true" in q,
            is_ambiguous=is_ambiguous
        )
    
class AmbiguityDetectorAgent:
    """
    Detects if query refers to multiple entities.
    """


    def is_ambiguous(self, query: str, expansions: List[str]) -> bool:
        q_words = set(query.lower().split())

        # if query is very short and expansions are many → ambiguous
        if len(q_words) <= 3 and len(expansions) >= 3:
            return True

        # if expansions are different proper names
        unique_tokens = set(e.lower() for e in expansions)
        return len(unique_tokens) >= 3
# ---------------------------------------------------------
# AGENT 4: QUERY DECOMPOSER
# ---------------------------------------------------------
class QueryDecomposerAgent:
    def decompose(self, query: str) -> List[str]:

        if " or " in query.lower():

            parts = query.split(" or ")

            # extract base question WITHOUT entities
            base = re.sub(r"which.*first", "", query, flags=re.IGNORECASE)

            # clean entities
            entities = []
            for p in parts:
                p = re.sub(r"which.*first", "", p, flags=re.IGNORECASE)
                p = p.strip(" ?")
                entities.append(p)

            # 🔥 KEY IDEA:
            # reuse full context instead of guessing type
            return [
                f"{base} {entities[0]}".strip(),
                f"{base} {entities[1]}".strip()
            ]

        return [query]

# ---------------------------------------------------------
# AGENT 5: QUERY EXPANSION
# ---------------------------------------------------------
class Expansion(BaseModel):
    expansions: List[str]

exp_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 5 concise search expansions."),
    ("human", "Query: {q}\nReturn JSON {{expansions: []}}")
])
exp_parser = PydanticOutputParser(pydantic_object=Expansion)

# ---------------------------------------------------------
# ✅ IMPROVED QUERY EXPANSION AGENT (DROP-IN REPLACEMENT)
# ---------------------------------------------------------

class QueryExpansionAgent:
    def expand(self, query: str) -> List[str]:
        """
        High-quality expansion:
        - Preserves entities
        - Avoids useless question forms
        - Produces search-optimized queries
        """

        prompt = f"""
You are a search query optimization expert.

Given a user question, generate EXACTLY 3 high-quality search queries.

STRICT RULES:
- Keep entity names EXACT (do NOT modify names)
- DO NOT generate questions like "Did X", "Is X", "Are X"
- DO NOT generate incomplete phrases
- Focus on factual retrieval
- Keep queries concise (3–8 words)
- Each query must be meaningful for Google/Wikipedia search

GOOD EXAMPLES:
Q: When was Albert Einstein born?
→ Albert Einstein birth year
→ Albert Einstein date of birth
→ when was Albert Einstein born

Q: Which magazine was started first Arthur's Magazine or First for Women?
→ Arthur's Magazine founding year
→ First for Women magazine start date
→ Arthur's Magazine vs First for Women timeline

NOW GENERATE:

Query: {query}

Return ONLY a JSON list:
["query1", "query2", "query3"]
"""

        raw = call_llm(prompt)

        # -------------------------------------------------
        # ✅ SAFE PARSE
        # -------------------------------------------------
        try:
            expansions = json.loads(re.search(r"\[.*\]", raw, re.DOTALL).group())
        except:
            expansions = [query]

        # -------------------------------------------------
        # ✅ CLEANING + FILTERING
        # -------------------------------------------------
        cleaned = []

        for e in expansions:
            e = normalize_query(e)

            # ❌ remove bad patterns
            if any(e.lower().startswith(w) for w in ["did", "are", "do", "is", "was", "were"]):
                continue

            # ❌ too short / junk
            if len(e.split()) < 2:
                continue

            # ❌ repetitive junk
            if e.lower() == query.lower():
                continue

            cleaned.append(e)

        # -------------------------------------------------
        # ✅ FALLBACK SAFETY
        # -------------------------------------------------
        if not cleaned:
            cleaned = [query]

        # limit to top 3
        cleaned = cleaned[:3]

        print(f"🧩 Improved expansions: {cleaned}")

        return cleaned
# ---------------------------------------------------------
# AGENT 6: HYPOTHESIS AGENT
# ---------------------------------------------------------
# ---------------------------------------------------------
# ✅ IMPROVED HYPOTHESIS AGENT
# ---------------------------------------------------------

class HypothesisAgent:
    def generate(self, query: str) -> List[str]:
        """
        Generate answer-oriented hypotheses to guide retrieval.
        """

        prompt = f"""
You are an expert QA system.

Given a question, generate 3 possible answer hypotheses.
These should look like factual statements that might appear in documents.

RULES:
- Convert question → statement form
- Include possible answer patterns (dates, names, locations)
- Keep concise
- Do NOT hallucinate unknown facts — just plausible structures

EXAMPLES:

Q: When was Albert Einstein born?
→ Albert Einstein was born in 1879
→ Albert Einstein birth year 1879
→ Albert Einstein date of birth April 1879

Q: Who directed Doctor Strange?
→ Doctor Strange was directed by Scott Derrickson
→ Scott Derrickson directed Doctor Strange
→ Director of Doctor Strange is Scott Derrickson

Q: What is the capital of France?
→ Paris is the capital of France
→ France capital city Paris
→ The capital of France is Paris

NOW DO:

Question: {query}

Return ONLY a JSON list:
["hypothesis1", "hypothesis2", "hypothesis3"]
"""

        raw = call_llm(prompt)

        # -------------------------------------------------
        # ✅ SAFE PARSE
        # -------------------------------------------------
        try:
            hyps = json.loads(re.search(r"\[.*\]", raw, re.DOTALL).group())
        except:
            hyps = [query]

        # -------------------------------------------------
        # ✅ CLEANING
        # -------------------------------------------------
        cleaned = []

        for h in hyps:
            h = normalize_query(h)

            if len(h.split()) < 3:
                continue

            cleaned.append(h)

        if not cleaned:
            cleaned = [query]

        print(f"🧠 Hypotheses: {cleaned}")

        return cleaned

# ---------------------------------------------------------
# AGENT 7: RETRIEVER
# ---------------------------------------------------------
class RetrieverAgent:
    def retrieve(self, expansions: List[str]) -> List[Tuple[str, List[str]]]:
        results = []

        for e in expansions:
            try:
                clean_e = normalize_query(e)

                # 🔍 DEBUG (optional but useful)
                print(f"🔎 Retrieving for: {clean_e}")

                search_res = search_tool.run(clean_e)
                wiki_res = wiki_tool.run(clean_e)

                docs = [
                    str(search_res)[:500],
                    str(wiki_res)[:500]
                ]

            except Exception as ex:
                print(f"⚠️ Retrieval failed for {e}: {ex}")
                clean_e = normalize_query(e)
                docs = ["No results"]

            results.append((clean_e, docs))

        return results
class EntitySelectionAgent:
    """
    Selects best entity expansion based on semantic similarity.
    """

    def select_best(self, query: str, ranked_expansions: List[Tuple[str, float]], top_k=1):
        if not ranked_expansions:
            return []

        # If query is generic like "who is X"
        if len(query.split()) <= 3:
            return [ranked_expansions[0][0]]

        return [exp for exp, _ in ranked_expansions[:top_k]]
# ---------------------------------------------------------
# AGENT 8: HYBRID RANKER
# ---------------------------------------------------------
class HybridRankerAgent:
    def rank(self, query: str, docs: List[Tuple[str, List[str]]]):
        texts = [" ".join(d) for _, d in docs]
        tokenized = [word_tokenize(t.lower()) for t in texts]
        bm25 = BM25Okapi(tokenized)

        q_emb = semantic_model.encode(query)
        doc_embs = semantic_model.encode(texts)

        scores = []
        for i, (exp, _) in enumerate(docs):
            bm = bm25.get_scores(word_tokenize(exp.lower()))[i]
            sem = cosine_similarity([q_emb], [doc_embs[i]])[0][0]
            scores.append((exp, bm * 0.3 + sem * 0.7))

        return sorted(scores, key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------
# AGENT 9–11: SAFETY + CONTRADICTION
# ---------------------------------------------------------
class SkepticAgent:
    def check(self, text: str) -> bool:
        return not any(w in text.lower() for w in ["toxic", "danger", "death"])

class ContradictionAgent:
    def detect(self, docs: List[str]) -> bool:
        return any("however" in d.lower() for d in docs)

# ---------------------------------------------------------
# AGENT 12: ANSWER SYNTHESIZER
# ---------------------------------------------------------
class Answer(BaseModel):
    answer: str
    justification: str
    confidence: float

ans_prompt = ChatPromptTemplate.from_messages([
    (
        "You are a careful research assistant.\n\n"
        "Instructions:\n"
        "1. Identify all entities mentioned in the question.\n"
        "2. If the question is a comparison:\n"
        "   - Extract relevant values (e.g., dates, names) for EACH entity\n"
        "   - Compare them explicitly\n"
        "   - Return ONLY the correct entity as the answer\n"
        "3. If evidence exists, you MUST answer.\n"
        "4. Only say information is missing if truly absent.\n"
        "Use ONLY the provided context.\n"
        "Return ONLY valid JSON.\n\n"
        "Format exactly:\n"
        "{{\n"
        '  "answer": "...",\n'
        '  "justification": "...",\n'
        '  "confidence": 0.0\n'
        "}}"
    ),
    (
        "human",
        "Question: {q}\n\n"
        "Context documents:\n"
        "{ctx}\n\n"
        "Remember:\n"
        "- Answer in ONE short phrase(entity name, number or date)\n"
        "Do NOT include explanations in the answer field.\n"
        "Put reasoning only in justification.\n"
        "- Justification must reference the evidence\n"
        "- Return ONLY JSON"
    ),
])
ans_parser = PydanticOutputParser(pydantic_object=Answer)

class AnswerSynthesizerAgent:
    def answer(self, query: str, docs: List[str]) -> Answer:
        context = "\n\n".join(docs)[:5000]  # safe token window

        raw = call_llm(ans_prompt.format(q=query, ctx=context))

        # -------------------------------------------------
        # ✅ Try strict parse
        # -------------------------------------------------
        try:
            return ans_parser.parse(raw)

        except Exception:
            print("⚠️ JSON parse failed — using safe fallback")

            data = safe_json(raw)

            if data and "answer" in data:
                return Answer(
                    answer=str(data.get("answer", "")).strip(),
                    justification=str(data.get("justification", "")).strip(),
                    confidence=float(data.get("confidence", 0.5))
                )

            # final fallback (still structured)
            return Answer(
                answer=raw.strip(),
                justification="The answer was generated from the retrieved context, but structured justification could not be parsed.",
                confidence=0.4
            )
# ---------------------------------------------------------
# AGENT 13: VERIFIER
# ---------------------------------------------------------
class VerifierAgent:
    def verify(self, answer: str, docs: List[str]) -> float:
        overlap = set(answer.lower().split()) & set(" ".join(docs).lower().split())
        return len(overlap) / max(len(answer.split()), 1)

# ---------------------------------------------------------
# 🧠 FLEXIBLE ORCHESTRATOR (THE HEART)
# ---------------------------------------------------------
class FlexibleOrchestrator:
    def __init__(self):
        self.classifier = QueryClassifierAgent()
        self.decomposer = QueryDecomposerAgent()
        self.expander = QueryExpansionAgent()
        self.hypothesis = HypothesisAgent()
        self.retriever = RetrieverAgent()
        self.ranker = HybridRankerAgent()
        self.skeptic = SkepticAgent()
        self.contradiction = ContradictionAgent()
        self.answerer = AnswerSynthesizerAgent()
        self.verifier = VerifierAgent()
        self.ambiguity = AmbiguityDetectorAgent()
        self.entity_selector = EntitySelectionAgent()

    def run(self, query: str):
        profile = self.classifier.classify(query)

        print("\n🧠 Query Profile:", profile.model_dump())

        sub_queries = self.decomposer.decompose(query) if profile.requires_decomposition else [query]
        expansions = []

        if " or " in query.lower():
            expansions = sub_queries

        else:
            expansions = self.expander.expand(sub_queries[-1])

        # fallback safety
        if not expansions:
            expansions = [query]

        expansions = expansions[:3]
        retrieved = self.retriever.retrieve(expansions)
        # -------------------------------------------------
        # 🧠 Ambiguity-aware evidence selection
        # -------------------------------------------------

        ranked = self.ranker.rank(query, retrieved)
        retrieved_dict = dict(retrieved)

        TOP_K = 5 if profile.is_ambiguous else 3

        if profile.is_ambiguous:
            print("⚠️ Ambiguous query detected — aggregating multiple entities")

        selected_exps = [exp for exp, _ in ranked[:TOP_K]]

        # collect docs
        all_docs = []
        for exp in selected_exps:
            all_docs.extend(retrieved_dict.get(exp, []))

        # deduplicate
        unique_docs = list(dict.fromkeys(all_docs))

        if profile.requires_safety and not self.skeptic.check(" ".join(unique_docs)):
            return {"answer": "Unsafe query", "confidence": 0.0}

        answer = self.answerer.answer(query, unique_docs)
        verification = self.verifier.verify(answer.answer, unique_docs)

        save_tool.run(answer.model_dump_json())
        print("\n📄 Context preview:")
        for d in unique_docs[:2]:
            print("-", d[:120])

        return {
            "query": query,
            "answer": answer.answer,
            "justification": answer.justification,  
            "confidence": round(answer.confidence * verification, 2),
            "verification": verification,
            "orchestration": profile.model_dump()
        }

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    q = input("Ask your question: ")
    orchestrator = FlexibleOrchestrator()
    result = orchestrator.run(q)
    

    pretty_print_result(result)