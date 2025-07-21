import asyncio
import time
from collections import defaultdict
from typing import List
from ..services import LLMService

# --- Constants ---
TOPIC_SUMMARY_SYSTEM_PROMPT = "Your task is to provide a concise, comprehensive summary in Simplified Chinese for the following text chunks, which all belong to a single topic. Output only the summary itself, without any introductory phrases."
OVERALL_SUMMARY_SYSTEM_PROMPT = "Your task is to create a high-level, overarching summary in Simplified Chinese from the following topic summaries. The summary should capture the main themes of the entire document. Output only the summary itself."
SAFE_CONTEXT_SIZE = 20000  # Safe character limit for a single LLM prompt

# --- Rate Limiter ---
class RateLimiter:
    """A simple async rate limiter to control requests per minute."""
    def __init__(self, max_rpm: int):
        if max_rpm <= 0:
            self.delay = 0
        else:
            self.delay = 60.0 / max_rpm
        self.lock = asyncio.Lock()
        self.last_request_time = 0

    async def __aenter__(self):
        if self.delay == 0:
            return
        
        async with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)
            self.last_request_time = time.time()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

# --- Core Functions ---
async def _generate_single_summary(user_prompt: str, system_prompt: str, summarize_llm_service: LLMService, rate_limiter: RateLimiter) -> str:
    """Generates a summary for a single block of text, respecting the rate limit."""
    async with rate_limiter:
        return await summarize_llm_service.generate(user_prompt=user_prompt, system_prompt=system_prompt)

async def _summarize_map_reduce(
    chunks: List[str],
    summarize_llm_service: LLMService,
    rate_limiter: RateLimiter
) -> str:
    """
    Summarizes a list of text chunks using a map-reduce approach to handle large volumes of text.
    """
    print(f"  - Activating map-reduce for {len(chunks)} chunks...")
    
    # Map phase: Create and summarize "super chunks"
    intermediate_summaries = []
    super_chunk_tasks = []
    current_super_chunk = ""

    for chunk_text in chunks:
        if len(current_super_chunk) + len(chunk_text) > SAFE_CONTEXT_SIZE:
            if current_super_chunk:
                prompt = f"TEXT CHUNKS:\n---\n{current_super_chunk}"
                task = _generate_single_summary(prompt, TOPIC_SUMMARY_SYSTEM_PROMPT, summarize_llm_service, rate_limiter)
                super_chunk_tasks.append(task)
            current_super_chunk = chunk_text
        else:
            current_super_chunk += "\n\n" + chunk_text

    if current_super_chunk:
        prompt = f"TEXT CHUNKS:\n---\n{current_super_chunk}"
        task = _generate_single_summary(prompt, TOPIC_SUMMARY_SYSTEM_PROMPT, summarize_llm_service, rate_limiter)
        super_chunk_tasks.append(task)

    print(f"  - Map phase: Created {len(super_chunk_tasks)} intermediate summary tasks.")
    map_results = await asyncio.gather(*super_chunk_tasks, return_exceptions=True)
    valid_intermediate_summaries = [res for res in map_results if not isinstance(res, Exception)]

    if not valid_intermediate_summaries:
        return "Error: Failed to generate any intermediate summaries in map-reduce."

    # Reduce phase: Summarize the intermediate summaries
    print(f"  - Reduce phase: Summarizing {len(valid_intermediate_summaries)} intermediate summaries.")
    final_summary_text = "\n\n".join(valid_intermediate_summaries)
    final_prompt = f"TOPIC SUMMARIES:\n---\n{final_summary_text}"
    final_summary = await _generate_single_summary(final_prompt, OVERALL_SUMMARY_SYSTEM_PROMPT, summarize_llm_service, rate_limiter)
    
    return final_summary

async def generate_summaries(
    clustered_data: list[dict], 
    summarize_llm_service: LLMService,
    summarization_chunk_threshold: int,
    summarize_max_rpm: int
) -> dict:
    """
    Generates a two-level summary from clustered text data, with robust fallbacks and rate limiting.
    """
    rate_limiter = RateLimiter(summarize_max_rpm)
    
    clusters = defaultdict(list)
    noise_points = []
    for item in clustered_data:
        if item['cluster_id'] == -1:
            noise_points.append(item)
        else:
            clusters[item['cluster_id']].append(item)
    
    print(f"Found {len(clusters)} topics to summarize and {len(noise_points)} noise points.")

    # --- Step 1: Summarize each valid topic ---
    topic_summary_tasks = {}
    for topic_id, chunks in sorted(clusters.items()):
        chunk_texts = [chunk['text'] for chunk in chunks]
        if len(chunk_texts) > summarization_chunk_threshold:
            print(f"  - Topic {topic_id} has {len(chunk_texts)} chunks, using map-reduce.")
            task = _summarize_map_reduce(chunk_texts, summarize_llm_service, rate_limiter)
        else:
            print(f"  - Summarizing topic {topic_id} ({len(chunk_texts)} chunks) directly...")
            full_topic_text = "\n\n".join(chunk_texts)
            user_prompt = f"TEXT CHUNKS:\n---\n{full_topic_text}"
            task = _generate_single_summary(user_prompt, TOPIC_SUMMARY_SYSTEM_PROMPT, summarize_llm_service, rate_limiter)
        topic_summary_tasks[topic_id] = task

    topic_summary_results = await asyncio.gather(*topic_summary_tasks.values(), return_exceptions=True)
    
    topic_summaries = {}
    for topic_id, result in zip(topic_summary_tasks.keys(), topic_summary_results):
        if isinstance(result, Exception):
            print(f"  - Summarization for topic {topic_id} failed: {result}")
            topic_summaries[topic_id] = f"Error summarizing topic: {result}"
        else:
            topic_summaries[topic_id] = result
            print(f"  - Summary for topic {topic_id} generated.")

    # --- Step 2: Generate Overall Summary (with fallback) ---
    overall_summary = ""
    if not clusters:
        print("⚠️ No valid topics found. Activating fallback: summarizing all text chunks via map-reduce.")
        all_chunk_texts = [item['text'] for item in clustered_data]
        if all_chunk_texts:
            overall_summary = await _summarize_map_reduce(all_chunk_texts, summarize_llm_service, rate_limiter)
            print("✅ Fallback summary generated.")
        else:
            overall_summary = "Error: No text content available to summarize."
    else:
        print("Generating overarching summary from topic summaries...")
        all_summaries_text = "\n\n".join(topic_summaries.values())
        overall_user_prompt = f"TOPIC SUMMARIES:\n---\n{all_summaries_text}"
        overall_summary = await _generate_single_summary(overall_user_prompt, OVERALL_SUMMARY_SYSTEM_PROMPT, summarize_llm_service, rate_limiter)
        print("✅ Overarching summary generated.")

    # --- Step 3: Assemble Final Structure ---
    final_structure = {
        "overall_summary": overall_summary,
        "topics": [],
        "noise_points": [chunk for chunk in noise_points]
    }

    for topic_id, summary in sorted(topic_summaries.items()):
        final_structure["topics"].append({
            "topic_id": topic_id,
            "topic_summary": summary,
            "chunks": clusters[topic_id]
        })
        
    return final_structure