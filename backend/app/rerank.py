import json
import re
import numpy as np
from typing import List, Dict, Tuple
from .settings import settings


def _compute_semantic_similarity(query: str, context_text: str) -> float:
    """
    Compute a simple semantic similarity score between query and context.
    Uses word overlap and basic text matching as a fallback when embeddings aren't available.
    """
    query_words = set(query.lower().split())
    context_words = set(context_text.lower().split())
    
    if not query_words or not context_words:
        return 0.0
    
    # Jaccard similarity (intersection over union)
    intersection = len(query_words & context_words)
    union = len(query_words | context_words)
    
    if union == 0:
        return 0.0
    
    jaccard = intersection / union
    
    # Bonus for exact phrase matches
    query_lower = query.lower()
    context_lower = context_text.lower()
    if query_lower in context_lower:
        jaccard += 0.2
    
    return min(jaccard, 1.0)


def rerank(query: str, contexts: List[Dict], use_llm: bool = True) -> List[Dict]:
    """
    Rerank contexts based on relevance to the query.
    
    Args:
        query: The user's query string
        contexts: List of context dictionaries, each containing at least 'text' field
        use_llm: Whether to use LLM-based reranking (requires OpenAI API key)
    
    Returns:
        List of reranked contexts (most relevant first)
    """
    if not contexts:
        return contexts
    
    # If only one context, no need to rerank
    if len(contexts) == 1:
        return contexts
    
    # Try LLM-based reranking if enabled and OpenAI is available
    if use_llm and settings.llm_provider == "openai" and settings.openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            
            # Build context descriptions for the LLM
            context_texts = []
            newline = "\n"
            for i, ctx in enumerate(contexts):
                title = ctx.get('title', 'Unknown')
                section = ctx.get('section', '')
                text = ctx.get('text', '')[:500]  # Limit text length for efficiency
                section_line = f"Section: {section}{newline}" if section else ""
                context_texts.append(
                    f"[Context {i}]{newline}Title: {title}{newline}"
                    f"{section_line}"
                    f"Text: {text}{newline}"
                )
            
            contexts_text = newline.join(context_texts)
            prompt = f"""Given the following user query and a list of contexts, rank the contexts by their relevance to the query.

User Query: {query}

Contexts:
{contexts_text}

Please return ONLY a JSON array of indices (0-based) representing the order of contexts from most relevant to least relevant. 
For example, if Context 2 is most relevant, then Context 0, then Context 1, return: [2, 0, 1]

Return only the JSON array, nothing else:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that ranks document contexts by relevance. Always return a valid JSON array of indices."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON array from the response
            json_match = re.search(r'\[[\d,\s]+\]', result_text)
            if json_match:
                ranked_indices = json.loads(json_match.group())
                
                # Validate indices
                valid_indices = [i for i in ranked_indices if 0 <= i < len(contexts)]
                
                if len(valid_indices) == len(contexts):
                    # Reorder contexts based on LLM ranking
                    reranked = [contexts[i] for i in valid_indices]
                    return reranked
                else:
                    # If some indices are invalid, use valid ones and append missing
                    reranked = [contexts[i] for i in valid_indices]
                    included = set(valid_indices)
                    for i, ctx in enumerate(contexts):
                        if i not in included:
                            reranked.append(ctx)
                    return reranked
        except Exception as e:
            # Fall back to semantic similarity if LLM reranking fails
            print(f"⚠ LLM reranking failed: {e}, falling back to semantic similarity")
    
    # Fallback: Use semantic similarity scoring
    scored_contexts: List[Tuple[float, Dict]] = []
    for ctx in contexts:
        # Combine title, section, and text for scoring
        title = ctx.get('title', '')
        section = ctx.get('section', '')
        text = ctx.get('text', '')
        
        # Create a combined text for similarity computation
        combined_text = f"{title} {section} {text}".strip()
        
        score = _compute_semantic_similarity(query, combined_text)
        scored_contexts.append((score, ctx))
    
    # Sort by score (descending) and return contexts
    scored_contexts.sort(key=lambda x: x[0], reverse=True)
    return [ctx for score, ctx in scored_contexts]