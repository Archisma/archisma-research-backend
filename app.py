import os
import time
import pandas as pd

# ✅ NEW imports (safe, stdlib)
import glob

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# CrewAI + Tavily
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from tavily import TavilyClient


# =========================
# CONFIG
# =========================
BASE_DIR = "data"

PDF_PATH  = os.path.join(BASE_DIR, "aurora_systems_full_policy_manual.pdf")
DOCX_PATH = os.path.join(BASE_DIR, "aurora_systems_long_term_strategy.docx")
XLSX_PATH = os.path.join(BASE_DIR, "aurora_systems_operational_financials.xlsx")

# ✅ NEW dirs for Lineage + Incidents
LINEAGE_DIR  = os.path.join(BASE_DIR, "lineage")
INCIDENT_DIR = os.path.join(BASE_DIR, "incidents")

# Render env vars (set in Render dashboard)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Set it in Render Environment Variables.")

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Archisma Research Backend", version="1.1")

# For Cloudflare Pages frontend calls:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchReq(BaseModel):
    query: str

class PublicSearchReq(BaseModel):
    topic: str

# ✅ NEW request models
class LineageReq(BaseModel):
    query: str

class ProdIssueReq(BaseModel):
    issue: str


# =========================
# 1) LOAD DOCUMENTS (Aurora)
# =========================
docs = []

# PDF
if os.path.exists(PDF_PATH):
    pdf_docs = PyPDFLoader(PDF_PATH).load()
    docs += pdf_docs
    print(f"✅ PDF loaded successfully: {os.path.basename(PDF_PATH)}")
    print(f"   Pages loaded from PDF: {len(pdf_docs)}")
else:
    print(f"❌ PDF not found: {PDF_PATH}")

# DOCX
if os.path.exists(DOCX_PATH):
    docx_docs = Docx2txtLoader(DOCX_PATH).load()
    docs += docx_docs
    print(f"✅ Word document loaded successfully: {os.path.basename(DOCX_PATH)}")
    print(f"   Sections loaded from Word: {len(docx_docs)}")
else:
    print(f"❌ Word document not found: {DOCX_PATH}")

# XLSX
if os.path.exists(XLSX_PATH):
    xls = pd.ExcelFile(XLSX_PATH)
    sheet_count = 0
    for sheet in xls.sheet_names:
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
        text = f"Excel Sheet: {sheet}\n\n" + df.to_string(index=False)
        docs.append(Document(page_content=text, metadata={"source": XLSX_PATH, "sheet": sheet}))
        sheet_count += 1
    print(f"✅ Excel file loaded successfully: {os.path.basename(XLSX_PATH)}")
    print(f"   Sheets loaded from Excel: {sheet_count}")
else:
    print(f"❌ Excel file not found: {XLSX_PATH}")

print("\n📄 FINAL LOAD SUMMARY")
print(f"✅ Total documents/pages loaded into RAG pipeline: {len(docs)}")

# =========================
# 2) SPLIT / CHUNK
# =========================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_documents = text_splitter.split_documents(docs)
print(f"\n✅ Chunking completed. Total chunks created: {len(split_documents)}")

# =========================
# 3) EMBEDDINGS (Render-safe)
# =========================
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GEMINI_API_KEY
)
print("✅ Embeddings ready: text-embedding-004")

# =========================
# 4) VECTOR STORE (FAISS) + RETRIEVER
# =========================
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✅ FAISS vector store created + Retriever ready (k=4)")

# =========================
# 5) LLM + 6) RAG CHAIN
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)
print("✅ Gemini LLM ready: gemini-2.5-flash")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
print("✅ RAG chain ready")

# =========================
# CrewAI + Tavily Tool (Public Search)
# =========================
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

class TavilySearchTool(BaseTool):
    name: str = "tavily_web_search"
    description: str = "Search the internet using Tavily and return short results with source URLs."

    def _run(self, query: str) -> str:
        if not tavily_client:
            return "Tavily is not configured (missing TAVILY_API_KEY)."

        result = tavily_client.search(
            query=query,
            max_results=6,
            include_answer=True,
            include_sources=True
        )

        answer = result.get("answer", "")
        sources = result.get("results", [])

        lines = []
        lines.append("WEB ANSWER:")
        lines.append(answer if answer else "(no answer returned)")
        lines.append("\nSOURCES:")
        for i, s in enumerate(sources, start=1):
            lines.append(f"{i}) {s.get('title','No title')}\n   {s.get('url','')}")
        return "\n".join(lines).strip()

tavily_tool = TavilySearchTool()

def run_crewai_public(topic: str) -> str:
    """
    3-agent CrewAI: research -> analyze -> write (<=100 words).
    Based on your crewai.txt structure (max_iter=3, allow_delegation=False, 100-word cap).
    """
    research_agent = Agent(
        role="Web Researcher",
        goal="Gather accurate, up-to-date information with sources.",
        backstory="Expert researcher.",
        llm=llm,
        tools=[tavily_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    analysis_agent = Agent(
        role="Research Analyst",
        goal="Structure and sanity-check findings.",
        backstory="Turns notes into an outline.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    writer_agent = Agent(
        role="Layman-friendly Writer",
        goal="Explain simply and briefly.",
        backstory="Beginner-friendly writer.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    task_1 = Task(
        description=(
            f"Research: '{topic}'.\n"
            "Use web search tool and include sources.\n"
            "Include: what it is, why used, how it works (steps), common implementations."
        ),
        expected_output="Research notes with sources.",
        agent=research_agent
    )

    task_2 = Task(
        description=(
            f"Create a structured outline for '{topic}'.\n"
            "Include: definition, why it matters, step-by-step, limitations, simple example.\n"
            "Flag uncertainty explicitly."
        ),
        expected_output="Structured outline.",
        agent=analysis_agent,
        context=[task_1]
    )

    task_3 = Task(
        description=(
            f"Write the final answer about '{topic}' for a beginner.\n"
            "HARD RULES:\n"
            "- MAX 100 WORDS total\n"
            "- Plain language\n"
            "- Include ONE tiny example\n"
            "- No extra sections\n"
            "- Include 2-5 source URLs at the end (short)."
        ),
        expected_output="<=100 word beginner-friendly explanation.",
        agent=writer_agent,
        context=[task_1, task_2]
    )

    crew = Crew(
        agents=[research_agent, analysis_agent, writer_agent],
        tasks=[task_1, task_2, task_3],
        verbose=True
    )

    start = time.time()
    result = crew.kickoff()
    elapsed = time.time() - start

    text = str(result).strip()
    words = text.split()
    if len(words) > 100:
        text = " ".join(words[:100])

    text += f"\n\n(Time: {elapsed:.1f}s)"
    return text


# ============================================================
# ✅ NEW: Lineage Vector Index (data/lineage/*)
# ============================================================
def load_text_files_as_documents(folder_path: str, label: str) -> list:
    docs_local = []
    if not os.path.isdir(folder_path):
        print(f"⚠️ Folder missing: {folder_path}")
        return docs_local

    for p in glob.glob(os.path.join(folder_path, "*.*")):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            docs_local.append(Document(
                page_content=f"{label}_FILE: {os.path.basename(p)}\n\n{content}",
                metadata={"source": p, "type": label}
            ))
        except Exception as e:
            print(f"⚠️ Failed reading {label} file {p}: {e}")
    print(f"✅ Loaded {len(docs_local)} {label} files from {folder_path}")
    return docs_local

lineage_docs = load_text_files_as_documents(LINEAGE_DIR, "LINEAGE")
lineage_store = FAISS.from_documents(lineage_docs, embeddings) if lineage_docs else None
lineage_retriever = lineage_store.as_retriever(search_kwargs={"k": 5}) if lineage_store else None

# ✅ UPDATED: Force Markdown TABLE output for Lineage (so frontend can render it as a table)
lineage_prompt = ChatPromptTemplate.from_template("""
You are a data lineage assistant.
Use ONLY the lineage context provided.

Return your answer STRICTLY as a MARKDOWN TABLE with exactly 2 columns:
| Section | Details |

And exactly these six rows in this order:
1) Sources (feeds/tables)
2) Transformations (step-by-step)
3) Targets (tables/metrics)
4) Downstream Consumers (if any)
5) Notes / Assumptions
6) Evidence (artifact filenames you used)

Rules:
- Do NOT write anything outside the table.
- Details cell should use bullet points separated using <br> so it stays readable inside one table cell.
- Evidence must list the exact artifact filenames.

Context:
{context}

Question:
{question}
""")

# ============================================================
# ✅ NEW: Incident/RCA Vector Index (data/incidents/*.md)
# ============================================================
incident_docs = load_text_files_as_documents(INCIDENT_DIR, "INCIDENT")
incident_store = FAISS.from_documents(incident_docs, embeddings) if incident_docs else None

# distance threshold: smaller = better match. Adjust if needed.
INCIDENT_MAX_DISTANCE = float(os.environ.get("INCIDENT_MAX_DISTANCE", "0.90"))

incident_prompt = ChatPromptTemplate.from_template("""
You are a production support assistant.
Use ONLY the internal incident/RCA context provided.

Return in this format:
- Closest Historical Match (incident id / title)
- Likely Root Cause
- Proven Resolution Steps (numbered)
- Validation Checklist
- Evidence (incident filenames used)

Context:
{context}

Issue:
{question}
""")

def try_internal_incident_match(issue_text: str) -> tuple[str | None, float | None]:
    if not incident_store:
        return None, None

    pairs = incident_store.similarity_search_with_score(issue_text, k=3)
    if not pairs:
        return None, None

    best_doc, best_score = pairs[0]
    # If best_score too large, treat as "no match"
    if best_score is None or best_score > INCIDENT_MAX_DISTANCE:
        return None, best_score

    ctx = "\n\n---\n\n".join([d.page_content for (d, s) in pairs])
    resp = (incident_prompt | llm).invoke({"context": ctx, "question": issue_text})
    return resp.content, best_score


# =========================
# ENDPOINTS
# =========================
@app.get("/")
def root():
    return {"ok": True, "message": "Archisma Research Backend is running. Use /health, /search, /public_search, /lineage, /prod_issue"}

@app.get("/health")
def health():
    return {
        "ok": True,
        "docs_loaded": len(docs),
        "chunks": len(split_documents),
        "tavily_configured": bool(TAVILY_API_KEY),
        "lineage_files_loaded": len(lineage_docs),
        "incident_files_loaded": len(incident_docs),
    }

@app.post("/search")
def search(req: SearchReq):
    try:
        resp = rag_chain.invoke(req.query)
        return {"answer": resp.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/public_search")
def public_search(req: PublicSearchReq):
    try:
        if not TAVILY_API_KEY:
            return {"answer": "Public search is not configured. Missing TAVILY_API_KEY on the server."}
        return {"answer": run_crewai_public(req.topic)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ NEW: Lineage endpoint
@app.post("/lineage")
def lineage(req: LineageReq):
    try:
        if not lineage_retriever:
            raise HTTPException(
                status_code=500,
                detail="Lineage artifacts not loaded. Create data/lineage and add dummy sql/json/csv files."
            )

        chain = (
            {"context": lineage_retriever, "question": RunnablePassthrough()}
            | lineage_prompt
            | llm
        )
        resp = chain.invoke(req.query)
        return {"answer": resp.content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ NEW: Production Issue endpoint (private-first + public fallback)
@app.post("/prod_issue")
def prod_issue(req: ProdIssueReq):
    try:
        issue_text = (req.issue or "").strip()
        if not issue_text:
            return {"answer": "Please paste an error / stack trace / log text."}

        # 1) Private-first: internal incident match
        internal_answer, score = try_internal_incident_match(issue_text)
        if internal_answer:
            return {"answer": f"{internal_answer}\n\n(Internal match distance: {score:.3f})"}

        # 2) Fallback to public (your existing agent)
        if not TAVILY_API_KEY:
            return {"answer": f"No strong internal match found (distance: {score}). Public fallback is unavailable because TAVILY_API_KEY is missing."}

        web_answer = run_crewai_public(f"Help troubleshoot this production issue:\n\n{issue_text}")
        return {"answer": f"No strong internal match found (distance: {score}).\n\nPublic fallback:\n{web_answer}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
