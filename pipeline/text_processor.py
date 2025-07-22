import asyncio
import re
import time
from typing import List
from astrbot.api import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..services import EmbeddingService, LLMService
from .utils import RateLimiter

# --- Prompts ---
TEXT_REPAIR_SYSTEM_PROMPT = """You are an expert text editor. Your task is to "purify" a given text chunk by extracting valuable information and discarding noise.

**Step 1: Purify the Content**
- Read the entire text chunk.
- Identify and **remove** all worthless parts. Worthless content includes: UI navigation text ("click here", "edit"), metadata (version tables, author lists), **lists of topics or headings without corresponding explanatory text**, lists of links, fragmented sentences, ads, etc.
- Keep only the substantive, valuable content.

**Step 2: Process the Purified Content**
- After purification, evaluate what remains.
- **If the remaining text is empty or meaningless (e.g., only contains headings)**: Your ONLY output should be the tag `<discard_chunk />`.
- **If valuable text remains**:
    1.  **Analyze Coherence**: Determine if the purified text contains one single, coherent topic or multiple distinct topics.
    2.  **Process the Text**:
        *   **If coherent (one topic)**: Repair grammar, fix formatting, and translate the entire text into Simplified Chinese. Enclose the final result in a single `<repaired_text>` tag.
        *   **If multiple topics**:
            a. Split the text into smaller, semantically coherent sub-chunks.
            b. For each sub-chunk, repair and translate it into Simplified Chinese.
            c. Enclose EACH final, translated sub-chunk in its own `<repaired_text>` tag.

**Summary of Your Output Rules:**
- If the chunk is entirely worthless after purification, output only: `<discard_chunk />`
- If the chunk has valuable parts, output the purified, repaired, and translated parts in one or more `<repaired_text>...</repaired_text>` tags.
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
            
            if '<discard_chunk />' in response:
                return []  # Signal to discard this chunk

            # More robust regex to handle potential LLM formatting errors (spaces, newlines in tags)
            matches = re.findall(r'<\s*repaired_text\s*>\s*(.*?)\s*<\s*/\s*repaired_text\s*>', response, re.DOTALL)
            
            if matches:
                # Further cleaning to ensure no empty strings are returned
                return [m.strip() for m in matches if m.strip()]
            else:
                logger.warning(f"  - LLM response for chunk was not a discard and did not contain valid tags. Attempt {attempt + 1}/{max_retries + 1}. Assuming it's a discard.")
                return [] # If no valid tags and not explicitly discarded, discard it to be safe.
        except Exception as e:
            logger.warning(f"  - LLM call failed on attempt {attempt + 1}/{max_retries + 1}. Error: {str(e)}")
    
    logger.error(f"  - Failed to process chunk after {max_retries + 1} attempts. Using original text.")
    return [chunk]

async def process_text_and_embed(
    text: str,
    repair_llm_service: LLMService,
    embedding_service: EmbeddingService,
    use_llm_repair: bool,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    repair_max_rpm: int = 60,
    embed_chunks: bool = True
) -> list[dict]:
    """
    Splits text into chunks, optionally repairs them, and optionally generates embeddings.
    """
    if not text:
        logger.warning("No text found to process. Aborting.")
        return []

    logger.info(f"Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Generated {len(chunks)} original chunks.")

    if use_llm_repair:
        logger.info("\n--- Starting Text Repair, Translation, and Re-chunking Step ---")
        rate_limiter = RateLimiter(repair_max_rpm)
        tasks = [_repair_and_translate_chunk_with_retry(chunk, repair_llm_service, rate_limiter) for chunk in chunks]
        repaired_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_chunks = []
        for i, result in enumerate(repaired_results):
            if isinstance(result, Exception):
                logger.warning(f"  - Processing for chunk {i} generated an exception: {str(result)}. Falling back to original.")
                final_chunks.append(chunks[i])
            else:
                final_chunks.extend(result)
        
        logger.info(f"--- Text Processing Step Complete: {len(chunks)} original chunks became {len(final_chunks)} final chunks. ---\n")
        chunks = final_chunks

    # Format base data structure
    processed_data = [{"chunk_id": i, "text": chunk, "embedding": None} for i, chunk in enumerate(chunks) if chunk]

    if not embed_chunks:
        logger.info("Skipping embedding generation as requested.")
        return processed_data

    logger.info("Generating embeddings for each chunk in parallel...")
    
    chunks_to_embed = [item['text'] for item in processed_data]
    embedding_tasks = [embedding_service.get_embedding(chunk) for chunk in chunks_to_embed]
    embedding_results = await asyncio.gather(*embedding_tasks, return_exceptions=True)

    final_processed_data = []
    for i, result in enumerate(embedding_results):
        if isinstance(result, Exception):
            logger.error(f"    Embedding for chunk {processed_data[i]['chunk_id']} generated an exception: {str(result)}")
        else:
            if result:
                processed_data[i]['embedding'] = result
                final_processed_data.append(processed_data[i])

    final_processed_data.sort(key=lambda x: x['chunk_id'])
    logger.info("Text processing and embedding complete.")
    return final_processed_data