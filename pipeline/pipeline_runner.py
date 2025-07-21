import asyncio
from . import content_extractor, text_processor, clusterer, summarizer
from ..services import LLMService, EmbeddingService

async def run_pipeline(
    url: str,
    repair_llm_service: LLMService,
    summarize_llm_service: LLMService,
    embedding_service: EmbeddingService,
    use_llm_repair: bool,
    use_clustering_summary: bool,
    debug_mode: bool = False,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    summarization_chunk_threshold: int = 10,
    summarize_max_rpm: int = 20,
    repair_max_rpm: int = 60
) -> dict | None:
    """
    Executes the full URL-to-knowledge-base pipeline in memory.
    """
    print("="*80)
    print(f"🚀 Starting Knowledge Base Pipeline for URL: {url}")
    print("="*80)

    # --- Step 1: Content Extraction ---
    print("\n[Step 1/4] Extracting content...")
    extracted_content = await content_extractor.extract_content_from_url(url, debug_mode=debug_mode)
    if not extracted_content:
        print("❌ Pipeline failed at content extraction step. Aborting.")
        return None
    print(f"✅ Content extraction complete for title: {extracted_content.title}")

    # --- Step 2: Text Processing and Embedding ---
    print("\n[Step 2/4] Processing text and generating embeddings...")
    processed_data = await text_processor.process_text_and_embed(
        text=extracted_content.text,
        repair_llm_service=repair_llm_service,
        embedding_service=embedding_service,
        use_llm_repair=use_llm_repair,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        repair_max_rpm=repair_max_rpm
    )
    if not processed_data:
        print("❌ Pipeline failed at text processing step. Aborting.")
        return None
    print(f"✅ Text processing complete. Generated {len(processed_data)} chunks.")

    if not use_clustering_summary:
        print("\n✅ Pipeline finished without clustering and summarization as requested.")
        print("="*80)
        return {"processed_chunks": processed_data}

    # --- Step 3: Clustering ---
    print("\n[Step 3/4] Clustering text chunks...")
    # Note: hdbscan is synchronous. For a fully async plugin, this should be
    # run in a thread pool executor to avoid blocking the event loop.
    loop = asyncio.get_running_loop()
    clustered_data = await loop.run_in_executor(None, clusterer.cluster_embeddings, processed_data)
    if not clustered_data:
        print("❌ Pipeline failed at clustering step. Aborting.")
        return None
    print(f"✅ Clustering complete.")

    # --- Step 4: Summarization ---
    print("\n[Step 4/4] Generating final hierarchical summary...")
    summary_data = await summarizer.generate_summaries(
        clustered_data=clustered_data,
        summarize_llm_service=summarize_llm_service,
        summarization_chunk_threshold=summarization_chunk_threshold,
        summarize_max_rpm=summarize_max_rpm
    )
    if not summary_data:
        print("❌ Pipeline failed at summarization step. Aborting.")
        return None
    print(f"✅ Summarization complete.")

    print("\n🎉 Pipeline execution finished successfully!")
    print("="*80)
    return summary_data