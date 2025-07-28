import asyncio
import json
import sys
import uuid
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
from quart import request, jsonify

from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, AstrBotConfig
from playwright.async_api import async_playwright

from .services import LLMService, EmbeddingService
from .pipeline import pipeline_runner


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle numpy data types during serialization.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


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
        self.tasks = {}

    async def initialize(self):
        """初始化服务并注册 Web API。"""
        logger.info("Initializing URL to Knowledge Base Plugin...")
        
        await self._check_and_install_playwright()

        # 注册 Web API
        self.context.register_web_api(
            route="/url_2_kb/add",
            view_handler=self.handle_url_request,
            methods=["POST"],
            desc="接收 URL 并启动知识库处理流水线"
        )
        self.context.register_web_api(
            route="/url_2_kb/status",
            view_handler=self.get_task_status,
            methods=["POST"],
            desc="查询指定任务的状态和结果"
        )
        logger.info("URL to Knowledge Base Plugin initialized and API endpoints registered.")

    async def handle_url_request(self):
        """处理来自 Web API 的请求，创建任务并返回 task_id。"""
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = await request.get_json()
        url = data.get("url")
        if not url:
            return jsonify({"error": "Missing 'url' parameter"}), 400

        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {"status": "pending", "url": url}
        
        logger.info(f"Created task {task_id} for URL: {url}")

        # 异步运行流水线，避免阻塞
        asyncio.create_task(self.run_pipeline_task(
            task_id=task_id,
            url=url,
            data=data
        ))

        return jsonify({
            "status": "accepted",
            "message": "Task accepted for processing.",
            "task_id": task_id
        }), 202

    async def run_pipeline_task(self, task_id: str, url: str, data: dict):
        """包装流水线运行，更新任务状态和结果。"""
        self.tasks[task_id]["status"] = "processing"
        try:
            # 从 API 参数获取任务配置
            use_llm_repair = data.get("use_llm_repair", False)
            use_clustering_summary = data.get("use_clustering_summary", True)
            repair_llm_provider_id = data.get("repair_llm_provider_id")
            summarize_llm_provider_id = data.get("summarize_llm_provider_id")
            embedding_provider_id = data.get("embedding_provider_id")
            chunk_size = data.get("chunk_size", 300)
            chunk_overlap = data.get("chunk_overlap", 50)

            # 从插件全局配置获取环境设置
            debug_mode = self.config.get("debug_mode", False)
            summarization_chunk_threshold = self.config.get("summarization_chunk_threshold", 10)
            summarize_max_rpm = self.config.get("summarize_max_rpm", 20)
            repair_max_rpm = self.config.get("repair_max_rpm", 60)

            # 根据传入的 provider_id 动态创建服务实例
            repair_llm_service = LLMService(self.context, repair_llm_provider_id)
            summarize_llm_service = LLMService(self.context, summarize_llm_provider_id)
            embedding_service = EmbeddingService(self.context, embedding_provider_id)

            result = await pipeline_runner.run_pipeline(
                url=url,
                repair_llm_service=repair_llm_service,
                summarize_llm_service=summarize_llm_service,
                embedding_service=embedding_service,
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
                logger.info(f"Task {task_id} for {url} completed successfully. Updating status to 'completed'.")
                # 使用自定义编码器将结果序列化为纯JSON兼容格式，然后再反序列化，以确保数据类型正确
                cleaned_result = json.loads(json.dumps(result, cls=NumpyJSONEncoder))
                self.tasks[task_id]["status"] = "completed"
                self.tasks[task_id]["result"] = cleaned_result
            else:
                logger.error(f"Task {task_id} for {url} failed during pipeline execution. Updating status to 'failed'.")
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error"] = "Pipeline returned no result."
        except Exception as e:
            logger.error(f"An unexpected error occurred in task {task_id} for {url}: {e}", exc_info=True)
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)

    async def get_task_status(self):
        """根据 task_id 返回任务状态和结果。"""
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = await request.get_json()
        task_id = data.get("task_id")
        if not task_id:
            return jsonify({"error": "Missing 'task_id' in request body"}), 400

        task = self.tasks.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404

        response_data = {"task_id": task_id, "status": task["status"], "url": task.get("url")}

        if task["status"] == "completed":
            response_data["result"] = task.get("result")
            # Log safely, without the large result object
            logger.debug(f"Returning 'completed' status for task {task_id} with result.")
        elif task["status"] == "failed":
            response_data["error"] = task.get("error")
            logger.debug(f"Returning 'failed' status for task {task_id}. Error: {response_data['error']}")
        else:
            # For 'pending' or 'processing', response_data is small and safe to log
            logger.debug(f"Returning status for task {task_id}: {response_data}")

        try:
            return jsonify(response_data)
        except Exception as e:
            # If jsonify fails, it's likely due to non-serializable data in the result.
            logger.error(f"CRITICAL: Failed to jsonify response for task {task_id}. This is the root cause of the client error. Error: {e}", exc_info=True)
            error_response = {
                "task_id": task_id,
                "status": "error", # This is what the client receives
                "url": task.get("url"),
                "error": f"Internal server error: Failed to serialize result. Check server logs for details. Type: {type(e).__name__}"
            }
            return jsonify(error_response), 500

    async def terminate(self):
        """插件停用时调用的方法。"""
        logger.info("URL to Knowledge Base Plugin terminated.")

    async def _check_and_install_playwright(self):
        """检查 Playwright 浏览器驱动是否存在，如果不存在则提示用户手动安装。"""
        logger.info("Checking for Playwright browser drivers...")
        try:
            async with async_playwright() as p:
                # 尝试启动浏览器，如果驱动不存在会抛出异常
                browser = await p.chromium.launch(headless=True)
                await browser.close()
            logger.info("Playwright browser drivers are already installed.")
        except Exception as e:
            # 捕获因驱动不存在而产生的异常
            if "Executable doesn't exist at" in str(e):
                error_message = (
                    "Playwright browser drivers not found. "
                    "Please install them manually by running the following command in your terminal:\n"
                    "python -m playwright install --with-deps"
                )
                logger.error(error_message)
                # 抛出运行时错误，中断插件加载
                raise RuntimeError(error_message)
            else:
                # 处理其他可能的 Playwright 相关异常
                logger.error(f"An unexpected error occurred while checking Playwright status: {e}", exc_info=True)
                raise e
