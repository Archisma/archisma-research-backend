Created Research  Project , One DIY Project. Name: Archisma Ghosal ; Contact No : 3023633094



Archisma Ghosal Research AI – Hybrid Public & Private Research Assistant
________________________________________
Project Overview
This project is a do-it-yourself research assistant prototype designed to support two distinct research workflows from a single web interface:
1.	Public Research – Uses an autonomous research agent (CrewAI) to perform real-time internet research and synthesis.
2.	Private Research – Uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions strictly from private enterprise documents.
The system is intentionally modular, secure, and extensible, enabling experimentation with modern AI agents, vector search, and LLM orchestration.
________________________________________
High-Level Architecture (Updated)
Frontend
•	Platform: Cloudflare Pages
•	Purpose:
o	Provides a clean UI with:
	Topic input
	Research category selector (Public / Private)
	Run Research button
	Dedicated output panel
o	Sends requests to a backend API
•	Security:
o	No API keys or secrets exposed
o	Stateless client
________________________________________
Backend API
•	Platform: Cloud server (Render / Railway / Fly.io)
•	Framework: FastAPI (Python)
•	Responsibilities:
o	Receives research requests
o	Routes requests based on research category
o	Orchestrates AI agents and RAG pipelines
o	Returns structured responses to the frontend
•	Security:
o	API keys stored only as environment variables
o	HTTPS-only communication
________________________________________
Public Research Flow (CrewAI-based)
Why CrewAI?
CrewAI enables multi-step, tool-augmented research rather than simple LLM completion. It is well suited for open-ended research tasks.
Flow
1.	User selects Public Research
2.	Backend initializes a CrewAI research agent
3.	Agent is equipped with:
o	A real web search tool (e.g., Tavily or Serper)
o	A Gemini LLM for reasoning and summarization
4.	The agent:
o	Searches the internet
o	Extracts relevant information
o	Synthesizes findings
o	Produces a structured answer with sources
5.	Response is returned to the frontend
Outcome
•	Internet-grounded answers
•	Tool-verified research (not hallucinated)
•	Suitable for exploratory or general knowledge queries
________________________________________
Confidential Research Flow (RAG-based)
Purpose
Enable secure, document-grounded research using enterprise-style internal documents.
Documents Used
•	Large policy PDF (20+ pages)
•	Large strategy Word document (20–30 pages)
•	Large Excel file (30+ rows of structured data)
Flow
1.	User selects Private Research
2.	Backend retrieves relevant document chunks using FAISS
3.	Retrieved context is passed to the LLM
4.	LLM answers strictly based on retrieved content
5.	No external internet access is used
Outcome
•	Answers are grounded in internal data
•	Reduced hallucination
•	Suitable for compliance, policy, and internal strategy questions
Input used for Demo:
•	pdf :aurora_systems_full_policy_manual.pdf
•	Word doc: aurora_systems_long_term_strategy.docx
•	Excel : aurora_systems_operational_financials.xlsx
________________________________________
Technologies Used (Updated)
Frontend
•	HTML / CSS / JavaScript
•	Cloudflare Pages
Backend
•	FastAPI
•	CrewAI (public research agent)
•	LangChain (document loading & orchestration)
•	FAISS (vector search)
•	Gemini 2.5 (LLM)
•	Tavily (web search tool for CrewAI)
________________________________________
Key Design Decisions
•	Separation of concerns: frontend, public research, and private research are isolated
•	Agent-based public search: CrewAI provides better research depth than a single prompt
•	RAG for private data: ensures accuracy and prevents hallucinations
•	Security-first: no secrets on the client side
•	Prototype-friendly: simple enough to build solo, realistic enough to scale
________________________________________
Final Outcome
This project demonstrates a modern hybrid research architecture that combines:
•	AI agents (CrewAI)
•	Tool-augmented web research
•	Private document retrieval
•	Large language models
•	Clean API-driven design
It serves as a strong foundation for learning, experimentation, and future production systems.

