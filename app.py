import os
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# =========================
# CONFIG
# =========================
BASE_DIR = "data"

PDF_PATH  = os.path.join(BASE_DIR, "aurora_systems_full_policy_manual.pdf")
DOCX_PATH = os.path.join(BASE_DIR, "aurora_systems_long_term_strategy.docx")
XLSX_PATH = os.path.join(BASE_DIR, "aurora_systems_operational_financials.xlsx")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Set this in Render env vars

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your Cloudflare Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchReq(BaseModel):
    query: str

# =========================
# LOAD DOCS (PDF + DOCX + XLSX)
# =========================
docs = []

# PDF
if os.path.exists(PDF_PATH):
    pdf_docs = PyPDFLoader(PDF_PATH).load()
    docs += pdf_docs
else:
    print(f"❌ PDF not found: {PDF_PATH}")

# DOCX
if os.path.exists(DOCX_PATH):
    docx_docs = Docx2txtLoader(DOCX_PATH).load()
    docs += docx_docs
else:
    print(f"❌ DOCX not found: {DOCX_PATH}")

# XLSX -> turn sheets into text docs
if os.path.exists(XLSX_PATH):
    xls = pd.ExcelFile(XLSX_PATH)
    for sheet in xls.sheet_names:
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
        text = f"Excel Sheet: {sheet}\n\n" + df.to_string(index=False)
        docs.append(Document(page_content=text, metadata={"source": XLSX_PATH, "sheet": sheet}))
else:
    print(f"❌ XLSX not found: {XLSX_PATH}")

print(f"✅ Loaded total docs/pages: {len(docs)}")

# =========================
# SPLIT / CHUNK
# =========================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_documents = text_splitter.split_documents(docs)
print(f"✅ Total chunks created: {len(split_documents)}")

# =========================
# EMBEDDINGS (LIGHTWEIGHT FOR RENDER 512MB)
# =========================
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not set. Set it in Render Environment Variables.")
else:
    print("✅ GEMINI_API_KEY found.")

embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GEMINI_API_KEY
)
print("✅ Gemini embeddings ready: text-embedding-004")

# =========================
# FAISS VECTORSTORE + RETRIEVER
# =========================
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✅ FAISS vector store ready (k=4)")

# =========================
# LLM (GEMINI CHAT)
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)
print("✅ Gemini LLM ready: gemini-1.5-flash")

# =========================
# RAG CHAIN
# =========================
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
# ENDPOINTS
# =========================
@app.get("/health")
def health():
    return {
        "ok": True,
        "docs_loaded": len(docs),
        "chunks": len(split_documents),
    }

@app.post("/search")
def search(req: SearchReq):
    resp = rag_chain.invoke(req.query)
    return {"answer": resp.content}
