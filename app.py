import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- paste your imports here (langchain loaders, pandas, etc.) ----
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import pandas as pd

app = FastAPI()

# allow your frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your Cloudflare domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchReq(BaseModel):
    query: str

# ---------- YOUR CODE STARTS HERE (same logic) ----------
BASE_DIR = "data"

PDF_PATH  = os.path.join(BASE_DIR, "aurora_systems_full_policy_manual.pdf")
DOCX_PATH = os.path.join(BASE_DIR, "aurora_systems_long_term_strategy.docx")
XLSX_PATH = os.path.join(BASE_DIR, "aurora_systems_operational_financials.xlsx")

docs = []

# PDF
pdf_docs = PyPDFLoader(PDF_PATH).load()
docs += pdf_docs

# DOCX
docx_docs = Docx2txtLoader(DOCX_PATH).load()
docs += docx_docs

# XLSX
xls = pd.ExcelFile(XLSX_PATH)
for sheet in xls.sheet_names:
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
    text = f"Excel Sheet: {sheet}\n\n" + df.to_string(index=False)
    docs.append(Document(page_content=text, metadata={"source": XLSX_PATH, "sheet": sheet}))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_documents = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # keep your key name
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)

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
# ---------- YOUR CODE ENDS HERE ----------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/search")
def search(req: SearchReq):
    resp = rag_chain.invoke(req.query)
    return {"answer": resp.content}

@app.get("/health")
def health():
    return {
        "ok": True,
        "docs_loaded": len(docs),
        "chunks": len(split_documents)
    }