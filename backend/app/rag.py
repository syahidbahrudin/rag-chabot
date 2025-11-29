import time, os, math, json, hashlib, re
from typing import List, Dict, Tuple
import numpy as np
from .settings import settings
from .ingest import chunk_text, doc_hash
from qdrant_client import QdrantClient, models as qm

# ---- Simple local embedder (deterministic) ----
def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in s.split()]

class LocalEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        # Hash-based repeatable pseudo-embedding
        h = hashlib.sha1(text.encode("utf-8")).digest()
        rng_seed = int.from_bytes(h[:8], "big") % (2**32-1)
        rng = np.random.default_rng(rng_seed)
        v = rng.standard_normal(self.dim).astype("float32")
        # L2 normalize
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

# ---- Vector store abstraction ----
class InMemoryStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict] = []
        self._hashes = set()

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(v.astype("float32"))
            self.meta.append(m)
            if h:
                self._hashes.add(h)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.vecs:
            return []
        A = np.vstack(self.vecs)  # [N, d]
        q = query.reshape(1, -1)  # [1, d]
        # cosine similarity
        sims = (A @ q.T).ravel() / (np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

class QdrantStore:
    def __init__(self, collection: str, dim: int = 384, url: str = "http://localhost:6333"):
        self.collection = collection
        self.dim = dim
        self.url = url
        print(f"🔌 Attempting to connect to Qdrant at {url}...")
        try:
            self.client = QdrantClient(url=url, timeout=10.0)
            # Test connection by getting collections list
            collections = self.client.get_collections()
            print(f"✓ Successfully connected to Qdrant at {url}")
            print(f"  Available collections: {[c.name for c in collections.collections]}")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant at {url}: {e}")
            raise
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collection_info = self.client.get_collection(self.collection)
            print(f"✓ Collection '{self.collection}' exists with {collection_info.points_count} points")
        except Exception as e:
            print(f"⚠ Collection '{self.collection}' not found, creating it...")
            try:
                self.client.recreate_collection(
                    collection_name=self.collection,
                    vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
                )
                print(f"✓ Created collection '{self.collection}' with dimension {self.dim}")
            except Exception as create_error:
                print(f"❌ Failed to create collection '{self.collection}': {create_error}")
                raise

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        points = []
        for i, (v, m) in enumerate(zip(vectors, metadatas)):
            point_id = i
            if m.get("id"):

                try:
                    point_id = int(m.get("id")[:16], 16)
                except (ValueError, TypeError):
                    point_id = i
            elif m.get("hash"):
                # Convert hex string to integer (use first 16 chars = 64 bits)
                try:
                    point_id = int(m.get("hash")[:16], 16)
                except (ValueError, TypeError):
                    point_id = i
            points.append(qm.PointStruct(id=point_id, vector=v.tolist(), payload=m))
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=k,
            score_threshold=0.0,  # Lower threshold to allow more relevant results
            with_payload=True
        )
        out = []
        for r in res:
            out.append((float(r.score), dict(r.payload)))
        return out

# ---- LLM provider ----
class StubLLM:
    def generate(self, query: str, contexts: List[Dict]) -> str:
        lines = [f"Answer (stub): Based on the following sources:"]
        for c in contexts:
            sec = c.get("section") or "Section"
            lines.append(f"- {c.get('title')} — {sec}")
        lines.append("Summary:")
        # naive summary of top contexts
        joined = " ".join([c.get("text", "") for c in contexts])
        lines.append(joined[:600] + ("..." if len(joined) > 600 else ""))
        return "\n".join(lines)


class OpenAILLM:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, query: str, contexts: List[Dict]) -> str:
        # Build context from retrieved chunks
        context_text = ""
        for i, c in enumerate(contexts, 1):
            title = c.get('title', 'Unknown')
            section = c.get('section', '')
            text = c.get('text', '')[:800]  # Increased context window
            context_text += f"\n[Source {i}]\nTitle: {title}\n"
            if section:
                context_text += f"Section: {section}\n"
            context_text += f"Content: {text}\n---\n"
        
        system_message = """You are a friendly, knowledgeable, and talkative company policy assistant. Your role is to help employees understand company policies, products, and procedures in a warm and engaging way.

CRITICAL INSTRUCTIONS:
- Only answer the question based on the information provided
- If the question is not related to the information provided, say "I'm sorry, I can't answer that question."

Guidelines for your responses:
- Format your response using Markdown for better readability
- Use headings (##, ###) to organize information
- Use bullet points (- or *) for lists
- Use **bold** for emphasis on important terms or key points
- Use `code` formatting for specific policy names, product codes, or technical terms
- Use numbered lists (1., 2., 3.) for step-by-step instructions
- Be conversational, friendly, and enthusiastic
- Provide detailed, comprehensive answers (not just brief summaries)
- Explain things clearly and thoroughly, as if you're having a helpful conversation
- Use natural language and be engaging - write as if you're explaining to a colleague
- Synthesize information from the sources into your own explanation
- Add helpful context, examples, and explanations to make the information more useful
- Be proactive in providing related information that might be helpful
- Use a warm, approachable tone throughout
- When referencing sources, do so naturally (e.g., "According to our [Title] policy..." or "As outlined in [Title]...") but don't copy the text

Remember: Your goal is to help the user understand the information, not to repeat the documents word-for-word. Explain, interpret, and synthesize the information in a natural, conversational way using Markdown formatting!"""

        user_message = f"""User Question: {query}

Relevant Information from Company Documents:
{context_text}

IMPORTANT: Use the information above as reference material, but DO NOT copy it verbatim. Instead:
- Don't just summarize the information, explain it in your own words
- Read and understand the information
- Synthesize it into your own words
- Explain it naturally as if you're a helpful colleague
- Add context and examples where helpful
- Make it conversational and easy to understand
- Only quote exact terms, dates, or policy names when necessary

Provide a detailed, friendly, and comprehensive answer that helps the user understand the information. Be thorough and conversational - explain things in your own words as if you're helping a colleague understand the policy or product."""
        
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.85,  # Higher temperature for more creative, interpretive responses
            max_tokens=1200   # Allow for longer, more detailed responses
        )
        return resp.choices[0].message.content

    def generate_stream(self, query: str, contexts: List[Dict]):
        """Generator that yields chunks of the response as they're generated"""
        # Build context from retrieved chunks
        context_text = ""
        for i, c in enumerate(contexts, 1):
            title = c.get('title', 'Unknown')
            section = c.get('section', '')
            text = c.get('text', '')[:800]  # Increased context window
            context_text += f"\n[Source {i}]\nTitle: {title}\n"
            if section:
                context_text += f"Section: {section}\n"
            context_text += f"Content: {text}\n---\n"
        
        system_message = """You are a friendly, knowledgeable, and talkative company policy assistant. Your role is to help employees understand company policies, products, and procedures in a warm and engaging way.

CRITICAL INSTRUCTIONS:
- Only answer the question based on the information provided
- If the question is not related to the information provided, say "I'm sorry, I can't answer that question."

Guidelines for your responses:
- Format your response using Markdown for better readability
- Use headings (##, ###) to organize information
- Use bullet points (- or *) for lists
- Use **bold** for emphasis on important terms or key points
- Use `code` formatting for specific policy names, product codes, or technical terms
- Use numbered lists (1., 2., 3.) for step-by-step instructions
- Be conversational, friendly, and enthusiastic
- Provide detailed, comprehensive answers (not just brief summaries)
- Explain things clearly and thoroughly, as if you're having a helpful conversation
- Use natural language and be engaging - write as if you're explaining to a colleague
- Synthesize information from the sources into your own explanation
- Add helpful context, examples, and explanations to make the information more useful
- Be proactive in providing related information that might be helpful
- Use a warm, approachable tone throughout
- When referencing sources, do so naturally (e.g., "According to our [Title] policy..." or "As outlined in [Title]...") but don't copy the text

Remember: Your goal is to help the user understand the information, not to repeat the documents word-for-word. Explain, interpret, and synthesize the information in a natural, conversational way using Markdown formatting!"""

        user_message = f"""User Question: {query}

Relevant Information from Company Documents:
{context_text}

IMPORTANT: Use the information above as reference material, but DO NOT copy it verbatim. Instead:
- Read and understand the information
- Synthesize it into your own words
- Explain it naturally as if you're a helpful colleague
- Add context and examples where helpful
- Make it conversational and easy to understand
- Only quote exact terms, dates, or policy names when necessary

Provide a detailed, friendly, and comprehensive answer that helps the user understand the information. Be thorough and conversational - explain things in your own words as if you're helping a colleague understand the policy or product."""
        print(context_text,'context_text')
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.95,
            max_tokens=1200,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

# ---- RAG Orchestrator & Metrics ----
class Metrics:
    def __init__(self):
        self.t_retrieval = []
        self.t_generation = []

    def add_retrieval(self, ms: float):
        self.t_retrieval.append(ms)

    def add_generation(self, ms: float):
        self.t_generation.append(ms)

    def summary(self) -> Dict:
        avg_r = sum(self.t_retrieval)/len(self.t_retrieval) if self.t_retrieval else 0.0
        avg_g = sum(self.t_generation)/len(self.t_generation) if self.t_generation else 0.0
        return {
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
        }

class RAGEngine:
    def __init__(self, collection: str = "policy_helper", dim: int = 384, url: str = "http://localhost:6333"):
        print(settings.llm_provider,'settings.llm_provider')
        print(settings.openai_api_key,'settings.openai_api_key')
        print(settings.vector_store,'settings.vector_store')
        print(settings.qdrant_url,'settings.qdrant_url')
        print(settings.collection_name,'settings.collection_name')
        print(settings.chunk_size,'settings.chunk_size')
        print(settings.chunk_overlap,'settings.chunk_overlap')
        print(settings.data_dir,'settings.data_dir')
        self.embedder = LocalEmbedder(dim=384)
        if settings.vector_store == "qdrant":
            try:
                self.store = QdrantStore(collection=settings.collection_name, dim=384, url=settings.qdrant_url)
            except Exception as e:
                self.store = InMemoryStore(dim=384)
        else:
            self.store = InMemoryStore(dim=384)

        if settings.llm_provider == "openai" and settings.openai_api_key:
            try:
                self.llm = OpenAILLM(api_key=settings.openai_api_key)
                self.llm_name = "openai:gpt-4o-mini"
            except Exception:
                self.llm = StubLLM()
                self.llm_name = "stub"
        else:
            self.llm = StubLLM()
            self.llm_name = "stub"

        self.metrics = Metrics()
        self._doc_titles = set()
        self._chunk_count = 0

    def ingest_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        vectors = []
        metas = []
        doc_titles_before = set(self._doc_titles)

        for ch in chunks:
            text = ch["text"]
            h = doc_hash(text)
            meta = {
                "id": h,
                "hash": h,
                "title": ch["title"],
                "section": ch.get("section"),
                "text": text,
            }
            v = self.embedder.embed(text)
            vectors.append(v)
            metas.append(meta)
            self._doc_titles.add(ch["title"])
            self._chunk_count += 1

        self.store.upsert(vectors, metas)
        return (len(self._doc_titles) - len(doc_titles_before), len(metas))

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        t0 = time.time()
        qv = self.embedder.embed(query)
        results = self.store.search(qv, k=k)
        print(results,'results')
        self.metrics.add_retrieval((time.time()-t0)*1000.0)
        for score, meta in results:
            meta['_score'] = score
        return [meta for score, meta in results]

    def generate(self, query: str, contexts: List[Dict]) -> str:
        t0 = time.time()
        answer = self.llm.generate(query, contexts)
        self.metrics.add_generation((time.time()-t0)*1000.0)
        return answer

    def generate_stream(self, query: str, contexts: List[Dict]):
        """Generator that streams the response as it's generated"""
        if hasattr(self.llm, 'generate_stream'):
            yield from self.llm.generate_stream(query, contexts)
        else:
            # Fallback: generate full response and yield it
            answer = self.llm.generate(query, contexts)
            yield answer

    def stats(self) -> Dict:
        m = self.metrics.summary()
        return {
            "total_docs": len(self._doc_titles),
            "total_chunks": self._chunk_count,
            "embedding_model": settings.embedding_model,
            "llm_model": self.llm_name,
            **m
        }

    def test_connection(self) -> Dict:
        """Test the connection to the vector store"""
        result = {
            "vector_store_type": type(self.store).__name__,
            "connected": False,
            "error": None
        }
        
        if isinstance(self.store, QdrantStore):
            try:
                # Try to get collections to test connection
                collections = self.store.client.get_collections()
                collection_info = self.store.client.get_collection(self.store.collection)
                result["connected"] = True
                result["url"] = self.store.url
                result["collection"] = self.store.collection
                result["points_count"] = collection_info.points_count
                result["available_collections"] = [c.name for c in collections.collections]
            except Exception as e:
                result["error"] = str(e)
        elif isinstance(self.store, InMemoryStore):
            result["connected"] = True
            result["points_count"] = len(self.store.vecs)
        
        return result

    def rerank(self, query: str, contexts: List[Dict]) -> List[Dict]:
        """Rerank the contexts based on the query using LLM"""
        if not contexts:
            return contexts
        
        # If using stub LLM, return contexts as-is
        if isinstance(self.llm, StubLLM):
            return contexts
        
        # Use LLM to rerank contexts
        try:
            # Build a prompt to score each context's relevance
            context_texts = []
            for i, ctx in enumerate(contexts):
                title = ctx.get('title', 'Unknown')
                section = ctx.get('section', '')
                text = ctx.get('text', '')[:500]  # Limit text length
                context_texts.append(f"[Context {i}]\nTitle: {title}\nSection: {section}\nText: {text}\n")
            
            prompt = f"""Given the following user query and a list of contexts, rank the contexts by their relevance to the query.

User Query: {query}

Contexts:
{chr(10).join(context_texts)}

Please return ONLY a JSON array of indices (0-based) representing the order of contexts from most relevant to least relevant. 
For example, if Context 2 is most relevant, then Context 0, then Context 1, return: [2, 0, 1]

Return only the JSON array, nothing else:"""
            
            # Use the LLM to generate the ranking
            if hasattr(self.llm, 'client'):
                # OpenAI LLM
                response = self.llm.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that ranks document contexts by relevance. Always return a valid JSON array of indices."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                result_text = response.choices[0].message.content.strip()
                
                # Try to parse the JSON array from the response
                # Extract JSON array from the response
                json_match = re.search(r'\[[\d,\s]+\]', result_text)
                if json_match:
                    ranked_indices = json.loads(json_match.group())
                    # Reorder contexts based on LLM ranking
                    reranked = [contexts[i] for i in ranked_indices if 0 <= i < len(contexts)]
                    # Add any contexts that weren't included in the ranking
                    included = set(ranked_indices)
                    for i, ctx in enumerate(contexts):
                        if i not in included:
                            reranked.append(ctx)
                    return reranked
            else:
                # Fallback: return original order
                return contexts
        except Exception as e:
            print(f"⚠ Error during reranking: {e}, returning original order")
            return contexts
        
        return contexts

# ---- Helpers ----
def build_chunks_from_docs(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    out = []
    for d in docs:
        for ch in chunk_text(d["text"], chunk_size, overlap):
            out.append({"title": d["title"], "section": d["section"], "text": ch})
    return out
