# ZoneMap Canada

**National Zoning & Land Use Data Platform**

An AI-powered platform that aggregates, standardizes, and makes searchable Canadian municipal zoning bylaws and land use regulations. Built for the Winter 2026 Hackathon by **Code Fellas**.

## The Problem

Across Canada's ~5,000 municipalities, zoning data is fragmented across thousands of websites, locked in PDFs, and written in inconsistent formats. Researchers, policymakers, and housing advocates have no comprehensive way to compare these regulations at scale.

## Our Solution

ZoneMap Canada uses an **agentic AI pipeline** to automatically discover, scrape, and extract structured zoning data from municipal websites. The platform provides:

- **Searchable database** of standardized zoning regulations
- **AI-powered extraction** using Google Gemini for structured data from PDFs and web pages
- **Comparison tools** to benchmark municipalities side-by-side
- **Restrictiveness scoring** to identify the most/least restrictive zoning policies
- **Interactive map** visualization with restrictiveness heatmap
- **Natural language chat** for asking questions about zoning regulations
- **Open data export** under CC BY 4.0 license

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| Web Scraping | Crawl4AI |
| PDF Extraction | pdfplumber |
| LLM (Extraction) | Gemini 2.5 Pro |
| LLM (Chat) | Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 2 Preview |
| Database | Supabase (PostgreSQL + PostGIS + pgvector) |
| Backend API | FastAPI |
| Frontend | Next.js + Tailwind CSS |
| Maps | Leaflet + OpenStreetMap |

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy .env.example to .env and fill in your API keys
cp .env.example .env

# Set up database tables
cd .. && PYTHONPATH=. python backend/setup_db.py

# Seed Toronto data
PYTHONPATH=. python backend/seed_toronto.py

# Run the API
PYTHONPATH=. uvicorn backend.api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install

# Create .env.local with:
# NEXT_PUBLIC_API_URL=http://localhost:8000/api

npm run dev
```

### Run Agent Pipeline (optional)

```bash
# Scrape additional municipalities
PYTHONPATH=. python backend/run_agent.py Mississauga Ottawa
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/municipalities` | List tracked municipalities |
| `GET /api/zones` | Query zones with filters |
| `GET /api/zones/nearby` | Geo-spatial zone search |
| `GET /api/zones/stats` | Aggregate statistics |
| `GET /api/compare` | Compare metrics across cities |
| `GET /api/compare/rankings` | Restrictiveness rankings |
| `GET /api/export?format=csv` | Export data as CSV |
| `GET /api/export?format=json` | Export data as JSON |
| `GET /api/review` | Items flagged for review |
| `POST /api/chat` | AI-powered Q&A |

## Data License

All extracted zoning data is available under **CC BY 4.0**. See [LICENSE](LICENSE).

## Team

**Code Fellas** - Winter 2026 Hackathon
