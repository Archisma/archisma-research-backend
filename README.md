 Name: Archisma Ghosal ; Contact No : 3023633094 ; EMP Id : 1384343 (TCS)


# Aurora Systems Research Facility
### Unified Intelligence Hub: Research • Lineage • Production Issue Resolution

**Author:** Okjishmagoshal  
**Contact:** (add your contact)

> **Demo Disclaimer:** Aurora Systems is a fictional/dummy enterprise dataset used solely for demonstration.  
> The architecture and workflows are generic and can be adapted to any organization.

---

## Live Demo (Cloudflare Pages)
https://aurora-archismaghosal-research-ai.pages.dev

## Repository
This repository hosts the **frontend** (static HTML/CSS/JS) deployed to Cloudflare Pages.

Backend is a FastAPI service (deployed separately) that provides:
- Public research via CrewAI + Tavily
- Confidential research via RAG (FAISS + internal docs)
- Lineage intelligence (RAG over mapping artifacts) *(if enabled)*
- Production issue intelligence (private-first incident retrieval with public fallback) *(if enabled)*

---

## Tabs / Features

### 1) Research
- **Public (Internet):** tool-verified research using Tavily + CrewAI + Gemini (short “baby researcher” output)
- **Confidential (Internal):** document-grounded answers via RAG (no internet)

### 2) Lineage Intelligence
- End-to-end lineage from mapping artifacts:
  source → transformations → target tables / reports

### 3) Production Issue Intelligence (Restricted)
- Intended for in-house use
- Private-first lookup (historical incidents/RCAs)
- Public fallback if issue is novel and not found internally
- The UI includes a **demo login gate** (credentials are not displayed in the UI)

> Note: The demo login gate is client-side and intended only for hackathon/demo purposes.  
> In a production setup, enforce access control on the backend (SSO/OAuth/Cloudflare Access) and protect endpoints server-side.

---

## Deployment (How it works)
1. Code is committed to GitHub
2. Cloudflare Pages is connected to this repo
3. Each push triggers automatic redeploy

---

## Security Notes
- Frontend is **stateless** and contains **no secrets**
- Backend API keys must be stored as environment variables (server-side)
- Use CORS allowlist and backend auth if deploying publicly

---

## License
Demo / prototype use.
