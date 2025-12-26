import os
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


# =========================
# CONFIG (Render/GitHub repo)
# =========================
# Put your files inside repo folder: data/
BASE_DIR = "data"

PDF_PATH  = os.path.join(BASE_DIR, "aurora_systems_full_policy_manual.pdf")
DOCX_PATH = os.path.join(BASE_DIR, "aurora_systems_long_term_strategy.docx")
XLSX_PATH = os.path.join(BASE_DIR, "aurora_systems_operational_financials.xlsx")

# IMPORTANT: Set this in Render -> Environment Variables
# Key: GEMINI_API_KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# =========================
# FASTAPI APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your Cloudflare domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchReq(BaseModel):
    query: str


# =========================
# 1) LOAD DOCUMENTS (same structure as your code)
# =========================
docs = []

# ---------- PDF ----------
if os.path.exists(PDF_PATH):
    pdf_docs = PyPDFLoader(PDF_PATH).load()
    docs += pdf_docs
    print(f"✅ PDF loaded successfully: {os.path.basename(PDF_PATH)}")
    print(f"   Pages loaded from PDF: {len(pdf_docs)}")
else:
    print(f"❌ PDF not found: {PDF_PATH}")

# ---------- WORD (DOCX) ----------
if os.path.exists(DOCX_PATH):
    docx_docs = Docx2txtLoader(DOCX_PATH).load()
    docs += docx_docs
    print(f"✅ Word document loaded successfully: {os.path.basename(DOCX_PATH)}")
    print(f"   Sections loaded from Word: {len(docx_docs)}")
else:
    print(f"❌ Word document not found: {DOCX_PATH}")

# ---------- EXCEL (XLSX) ----------
if os.path.exists(XLSX_PATH):
    xls = pd.ExcelFile(XLSX_PATH)
    sheet_count = 0
    for sheet in xls.sheet_names:
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
        text = f"Excel Sheet: {sheet}\n\n" + df.to_string(index=False)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": XLSX_PATH, "sheet": sheet}
            )
        )
        sheet_count += 1

    print(f"✅ Excel file loaded successfully: {os.path.basename(XLSX_PATH)}")
    print(f"   Sheets loaded from Excel: {sheet_count}")
else:
    print(f"❌ Excel file not found: {XLSX_PATH}")

print("\n📄 FINAL LOAD SUMMARY")
print(f"✅ Total documents/pages loaded into RAG pipeline: {len(docs)}")


# =========================
# 2) SPLIT / CHUNK (same as your code)
# =========================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_documents = text_splitter.split_documents(docs)
print(f"\n✅ Chunking completed. Total chunks created: {len(split_documents)}")


# =========================
# 3) EMBEDDINGS (Render-safe)
# - Your Colab used HuggingFaceEmbeddings (torch) => OOM on Render 512MB
# - Here we use Gemini embeddings to keep the SAME pipeline but fit memory.
# =========================
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY is missing. Set it in Render Environment Variables.")

embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GEMINI_API_KEY
)
print("✅ Embeddings ready: text-embedding-004")


# =========================
# 4) VECTOR STORE (FAISS) + RETRIEVER (same as your code)
# =========================
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✅ FAISS vector store created + Retriever ready (k=4)")


# =========================
# 5) LLM (Gemini) + 6) RAG CHAIN (your exact prompt and chain)
# =========================
# You used gemini-2.5-flash in Colab. If LangChain/model support differs on Render,
# switch to "gemini-1.5-flash" (only if needed).
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
    try:
        resp = rag_chain.invoke(req.query)
        return {"answer": resp.content}
    except Exception as e:
        # Return the real error message so you can debug quickly
        raise HTTPException(status_code=500, detail=str(e))
