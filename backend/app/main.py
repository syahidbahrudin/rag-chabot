from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
import json
from .models import IngestResponse, AskRequest, AskResponse, MetricsResponse, Citation, Chunk
from .settings import settings
from .ingest import load_documents
from .rag import RAGEngine, build_chunks_from_docs
from .guardrails import is_greeting, get_greeting_response

app = FastAPI(title="AI Policy & Product Helper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()

@app.get("/api/health")
def health():
    """Health check endpoint with vector store connection status"""
    health_status = {
        "status": "ok",
        "vector_store": engine.test_connection()
    }
    return health_status

@app.get("/api/metrics", response_model=MetricsResponse)
def metrics():
    s = engine.stats()
    return MetricsResponse(**s)

@app.post("/api/ingest", response_model=IngestResponse)
def ingest():
    docs = load_documents(settings.data_dir)
    chunks = build_chunks_from_docs(docs, settings.chunk_size, settings.chunk_overlap)
    new_docs, new_chunks = engine.ingest_chunks(chunks)
    return IngestResponse(indexed_docs=new_docs, indexed_chunks=new_chunks)

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Guardrail: Check if the query is a greeting
    if is_greeting(req.query):
        greeting_response = get_greeting_response()
        stats = engine.stats()
        return AskResponse(
            query=req.query,
            answer=greeting_response,
            citations=[],
            chunks=[],
            metrics={
                "retrieval_ms": 0.0,
                "generation_ms": 0.0,
            }
        )
    
    ctx = engine.retrieve(req.query, k=req.k or 4)
    answer = engine.generate(req.query, ctx)
    
    MIN_SCORE_THRESHOLD = 0.1
    MAX_CITATIONS = 3
    
    scored_docs = []
    for c in ctx:
        score = c.get('_score', 0.0)
        if score >= MIN_SCORE_THRESHOLD:
            scored_docs.append((score, c))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    seen_titles = set()
    filtered_citations = []
    for score, c in scored_docs:
        title = c.get("title")
        if title and title not in seen_titles and len(filtered_citations) < MAX_CITATIONS:
            seen_titles.add(title)
            filtered_citations.append(Citation(title=title, section=c.get("section")))
    
    chunks = [Chunk(title=c.get("title"), section=c.get("section"), text=c.get("text")) for c in ctx]
    stats = engine.stats()
    return AskResponse(
        query=req.query,
        answer=answer,
        citations=filtered_citations,
        chunks=chunks,
        metrics={
            "retrieval_ms": stats["avg_retrieval_latency_ms"],
            "generation_ms": stats["avg_generation_latency_ms"],
        }
    )

@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """Streaming endpoint for real-time responses"""
    # Guardrail: Check if the query is a greeting
    if is_greeting(req.query):
        greeting_response = get_greeting_response()
        def generate_greeting():
            # Send empty citations metadata
            yield f"data: {json.dumps({'type': 'metadata', 'citations': []})}\n\n"
            # Stream the greeting response in chunks to simulate streaming
            words = greeting_response.split()
            chunk_size = 3  # Send 3 words at a time
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size]) + " "
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(generate_greeting(), media_type="text/event-stream")
    
    ctx = engine.retrieve(req.query, k=req.k or 4)
    MIN_SCORE_THRESHOLD = 0.1
    MAX_CITATIONS = 3
    
    scored_docs = []
    for c in ctx:
        score = c.get('_score', 0.0)
        if score >= MIN_SCORE_THRESHOLD:
            scored_docs.append((score, c))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    seen_titles = set()
    unique_citations = []
    for score, c in scored_docs:
        title = c.get("title")
        if title and title not in seen_titles and len(unique_citations) < MAX_CITATIONS:
            seen_titles.add(title)
            unique_citations.append({
                "title": title,
                "section": c.get("section")
            })
    
    def generate():

        yield f"data: {json.dumps({'type': 'metadata', 'citations': unique_citations})}\n\n"
        

        full_text = ""
        for chunk in engine.generate_stream(req.query, ctx):
            full_text += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
        

        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
