import asyncio
from astrbot.api import logger
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
    logger.info("="*80)
    logger.info(f"🚀 Starting Knowledge Base Pipeline for URL: {url}")
    logger.info("="*80)

    # --- Step 1: Content Extraction ---
    logger.info("\n[Step 1/4] Extracting content...")
    extracted_content = await content_extractor.extract_content_from_url(url, debug_mode=debug_mode)
    if not extracted_content:
        logger.error("❌ Pipeline failed at content extraction step. Aborting.")
        return None
    logger.info(f"✅ Content extraction complete for title: {extracted_content.title}")

    # --- Step 2: Text Processing and Embedding ---
    logger.info("\n[Step 2/4] Processing text and generating embeddings...")
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
        logger.error("❌ Pipeline failed at text processing step. Aborting.")
        return None
    logger.info(f"✅ Text processing complete. Generated {len(processed_data)} chunks.")

    if not use_clustering_summary:
        logger.info("\n✅ Pipeline finished without clustering and summarization as requested.")
        # 清理并统一返回格式
        for chunk in processed_data:
            chunk.pop('embedding', None)
        
        # 按照最终格式返回，即使没有聚类和摘要
        final_result = {
            "overall_summary": "未执行聚类和摘要。",
            "topics": [],
            "noise_points": processed_data  # 所有块都视为未分类的点
        }
        logger.info("="*80)
        return final_result

    # --- Step 3: Clustering ---
    logger.info("\n[Step 3/4] Clustering text chunks...")
    # Note: hdbscan is synchronous. For a fully async plugin, this should be
    # run in a thread pool executor to avoid blocking the event loop.
    loop = asyncio.get_running_loop()
    clustered_data = await loop.run_in_executor(None, clusterer.cluster_embeddings, processed_data)
    if not clustered_data:
        logger.error("❌ Pipeline failed at clustering step. Aborting.")
        return None
    logger.info(f"✅ Clustering complete.")

    # --- Step 4: Summarization ---
    logger.info("\n[Step 4/4] Generating final hierarchical summary...")
    summary_data = await summarizer.generate_summaries(
        clustered_data=clustered_data,
        summarize_llm_service=summarize_llm_service,
        summarization_chunk_threshold=summarization_chunk_threshold,
        summarize_max_rpm=summarize_max_rpm
    )
    if not summary_data:
        logger.error("❌ Pipeline failed at summarization step. Aborting.")
        return None
    logger.info(f"✅ Summarization complete.")

    logger.info("\n🎉 Pipeline execution finished successfully!")
    logger.info("="*80)
    return summary_data