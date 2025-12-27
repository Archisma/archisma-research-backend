import os
import time
import pandas as pd

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

# Render env vars (set in Render dashboard)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Set it in Render Environment Variables.")

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Archisma Research Backend", version="1.0")

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
# CrewAI + Tavily Tool
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

    # hard timeout guard (simple)
    start = time.time()
    result = crew.kickoff()
    elapsed = time.time() - start

    text = str(result).strip()
    words = text.split()
    if len(words) > 100:
        text = " ".join(words[:100])

    text += f"\n\n(Time: {elapsed:.1f}s)"
    return text


# =========================
# ENDPOINTS
# =========================
@app.get("/")
def root():
    return {"ok": True, "message": "Archisma Research Backend is running. Use /health, /search, /public_search"}

@app.get("/health")
def health():
    return {
        "ok": True,
        "docs_loaded": len(docs),
        "chunks": len(split_documents),
        "tavily_configured": bool(TAVILY_API_KEY),
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
