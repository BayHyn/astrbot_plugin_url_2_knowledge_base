import asyncio
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from quart import request, jsonify

from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, AstrBotConfig
from playwright.async_api import async_playwright

from .services import LLMService, EmbeddingService
from .pipeline import pipeline_runner

@register(
    "url_2_knowledge_base", 
    "RC-CHN", 
    "通过 URL 提取内容，并经过处理、聚类和总结后，生成知识库文件。", 
    "v1.0.0"
)
class Url2KbPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.repair_llm_service: LLMService = None
        self.summarize_llm_service: LLMService = None
        self.embedding_service: EmbeddingService = None

    async def initialize(self):
        """初始化服务并注册 Web API。"""
        logger.info("Initializing URL to Knowledge Base Plugin...")
        
        # 根据用户配置初始化服务
        await self._check_and_install_playwright()
        self.repair_llm_service = LLMService(self.context, self.config.get("repair_llm_provider_id"))
        self.summarize_llm_service = LLMService(self.context, self.config.get("summarize_llm_provider_id"))
        self.embedding_service = EmbeddingService(self.context, self.config.get("embedding_provider_id"))

        # 注册 Web API
        self.context.register_web_api(
            route="/url_2_kb/add",
            view_handler=self.handle_url_request,
            methods=["POST"],
            desc="接收 URL 并启动知识库处理流水线"
        )
        logger.info("URL to Knowledge Base Plugin initialized and API endpoint registered.")

    async def handle_url_request(self):
        """处理来自 Web API 的请求。"""
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = await request.get_json()
        url = data.get("url")
        use_llm_repair = data.get("use_llm_repair", False)
        use_clustering_summary = data.get("use_clustering_summary", True)
        debug_mode = self.config.get("debug_mode", False)
        chunk_size = self.config.get("chunk_size", 300)
        chunk_overlap = self.config.get("chunk_overlap", 50)
        summarization_chunk_threshold = self.config.get("summarization_chunk_threshold", 10)
        summarize_max_rpm = self.config.get("summarize_max_rpm", 20)
        repair_max_rpm = self.config.get("repair_max_rpm", 60)

        if not url:
            return jsonify({"error": "Missing 'url' parameter"}), 400

        logger.info(f"Received request to process URL: {url}")

        # 异步运行流水线，避免阻塞
        asyncio.create_task(self.run_pipeline_task(
            url, use_llm_repair, use_clustering_summary, debug_mode,
            chunk_size, chunk_overlap, summarization_chunk_threshold,
            summarize_max_rpm, repair_max_rpm
        ))

        return jsonify({
            "status": "success",
            "message": "Pipeline started. Processing will continue in the background."
        }), 202

    async def run_pipeline_task(self, url, use_llm_repair, use_clustering_summary, debug_mode,
                                chunk_size, chunk_overlap, summarization_chunk_threshold,
                                summarize_max_rpm, repair_max_rpm):
        """包装流水线运行，以便于错误捕获。"""
        try:
            result = await pipeline_runner.run_pipeline(
                url=url,
                repair_llm_service=self.repair_llm_service,
                summarize_llm_service=self.summarize_llm_service,
                embedding_service=self.embedding_service,
                use_llm_repair=use_llm_repair,
                use_clustering_summary=use_clustering_summary,
                debug_mode=debug_mode,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                summarization_chunk_threshold=summarization_chunk_threshold,
                summarize_max_rpm=summarize_max_rpm,
                repair_max_rpm=repair_max_rpm
            )
            if result:
                logger.info(f"Pipeline for {url} completed successfully.")
                try:
                    data_dir = StarTools.get_data_dir()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    file_name = f"{timestamp}_{url_hash}.json"
                    file_path = data_dir / file_name
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=4)
                        
                    logger.info(f"Successfully saved knowledge base to: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to save result for {url}. Error: {e}", exc_info=True)
            else:
                logger.error(f"Pipeline for {url} failed.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in the pipeline for {url}: {e}", exc_info=True)

    async def terminate(self):
        """插件停用时调用的方法。"""
        logger.info("URL to Knowledge Base Plugin terminated.")

    async def _check_and_install_playwright(self):
        """检查 Playwright 浏览器驱动是否存在，如果不存在则尝试自动安装。"""
        logger.info("Checking for Playwright browser drivers...")
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                await browser.close()
            logger.info("Playwright browser drivers are already installed.")
        except Exception as e:
            if "Executable doesn't exist at" in str(e):
                logger.warning("Playwright browser drivers not found. Attempting to install automatically...")
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: subprocess.run(["python", "-m", "playwright", "install", "--with-deps"], check=True, capture_output=True, text=True)
                    )
                    logger.info("Playwright browser drivers installed successfully.")
                except subprocess.CalledProcessError as install_error:
                    logger.error(f"Failed to install Playwright browsers automatically. Please run 'python -m playwright install --with-deps' manually. Error: {install_error.stderr}")
                except Exception as install_e:
                    logger.error(f"An unexpected error occurred during automatic installation. Please run 'python -m playwright install --with-deps' manually. Error: {install_e}")
            else:
                logger.warning(f"An issue occurred while checking Playwright status: {e}")
