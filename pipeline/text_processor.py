import asyncio
import re
import time
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..services import EmbeddingService, LLMService

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

# --- Prompts ---
TEXT_REPAIR_SYSTEM_PROMPT = """You are an expert text processor. Your task is to analyze the given text chunk and improve it.

Follow these steps:
1.  **Analyze Coherence**: First, determine if the chunk contains one single, coherent topic or multiple distinct topics.
2.  **Process the Text**:
    *   **If the chunk is coherent (one topic)**: Repair any grammatical errors or formatting issues, then translate the entire repaired text into Simplified Chinese. Enclose the final result in a single `<repaired_text>` tag.
    *   **If the chunk contains multiple topics**:
        a. First, split the chunk into smaller, semantically coherent sub-chunks.
        b. For each sub-chunk, repair and translate it into Simplified Chinese.
        c. Enclose EACH of the final, translated sub-chunks in its own `<repaired_text>` tag.
3.  **Output Format**: Your entire output must ONLY be the `<repaired_text>` tags. Do not add any other text.

Example for a single topic:
<repaired_text>修复和翻译后的单一文本块</repaired_text>

Example for multiple topics:
<repaired_text>修复和翻译后的子块一</repaired_text>
<repaired_text>修复和翻译后的子块二</repaired_text>
"""

async def _repair_and_translate_chunk_with_retry(chunk: str, repair_llm_service: LLMService, rate_limiter: RateLimiter, max_retries: int = 2) -> List[str]:
    """
    Repairs, translates, and optionally re-chunks a single text chunk using the small LLM, with rate limiting.
    """
    user_prompt = f"Here is the text chunk to process:\n{chunk}"
    for attempt in range(max_retries + 1):
        try:
            async with rate_limiter:
                response = await repair_llm_service.generate(user_prompt=user_prompt, system_prompt=TEXT_REPAIR_SYSTEM_PROMPT)
            
            matches = re.findall(r'<repaired_text>(.*?)</repaired_text>', response, re.DOTALL)
            if matches:
                return [m.strip() for m in matches if m.strip()]
            else:
                print(f"  - Warning: LLM response for chunk did not contain valid tags. Attempt {attempt + 1}/{max_retries + 1}.")
        except Exception as e:
            print(f"  - Warning: LLM call failed on attempt {attempt + 1}/{max_retries + 1}. Error: {e}")
    
    print(f"  - Error: Failed to process chunk after {max_retries + 1} attempts. Using original text.")
    return [chunk]

async def process_text_and_embed(
    text: str,
    repair_llm_service: LLMService,
    embedding_service: EmbeddingService,
    use_llm_repair: bool,
    chunk_size: int = 1000, 
    chunk_overlap: int = 150,
    repair_max_rpm: int = 60
) -> list[dict]:
    """
    Splits text into chunks, optionally repairs them, and generates embeddings.
    """
    if not text:
        print("No text found to process. Aborting.")
        return []

    print(f"Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    print(f"Generated {len(chunks)} original chunks.")

    if use_llm_repair:
        print("\n--- Starting Text Repair, Translation, and Re-chunking Step ---")
        rate_limiter = RateLimiter(repair_max_rpm)
        tasks = [_repair_and_translate_chunk_with_retry(chunk, repair_llm_service, rate_limiter) for chunk in chunks]
        repaired_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_chunks = []
        for i, result in enumerate(repaired_results):
            if isinstance(result, Exception):
                print(f"  - Processing for chunk {i} generated an exception: {result}. Falling back to original.")
                final_chunks.append(chunks[i])
            else:
                final_chunks.extend(result)
        
        print(f"--- Text Processing Step Complete: {len(chunks)} original chunks became {len(final_chunks)} final chunks. ---\n")
        chunks = final_chunks

    processed_data = []
    print("Generating embeddings for each chunk in parallel...")
    
    # Note: Embedding generation might also need rate limiting if the provider has strict RPM limits.
    # For now, we assume it's less of a concern than LLM calls.
    embedding_tasks = [embedding_service.get_embedding(chunk) for chunk in chunks if chunk]
    embedding_results = await asyncio.gather(*embedding_tasks, return_exceptions=True)

    for i, result in enumerate(embedding_results):
        if isinstance(result, Exception):
            print(f"    Embedding for chunk {i} generated an exception: {result}")
        else:
            if result:
                processed_data.append({
                    "chunk_id": i,
                    "text": chunks[i],
                    "embedding": result
                })

    processed_data.sort(key=lambda x: x['chunk_id'])
    print("Text processing and embedding complete.")
    return processed_data