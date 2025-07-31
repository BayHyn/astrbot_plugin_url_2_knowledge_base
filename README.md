# URL to Knowledge Base 插件

**一个 AstrBot 插件，通过 URL 提取内容，并经过处理、聚类和总结后，生成知识库文件。**

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/RC-CHN/astrbot_plugin_url_2_knowledge_base)
[![License](https://img.shields.io/github/license/RC-CHN/astrbot_plugin_url_2_knowledge_base)](LICENSE)

## 🚀 功能

- **内容提取**: 从任意 URL 提取正文内容。
- **文本处理**: 自动清理和格式化提取的 HTML 内容。
- **智能分块**: 将长文本分割成语义完整的块。
- **LLM 文本修复 (可选)**: 利用大语言模型（LLM）纠正格式错误和乱码。
- **聚类总结 (可选)**: 对文本块进行聚类，并为每个类别生成摘要，快速掌握核心观点。
- **异步处理**: 任务在后台运行，不阻塞 AstrBot 主流程。
- **状态查询**: 提供 API 端点以查询任务的实时状态和结果。

## 🛠️ 安装与依赖

### 1. 安装插件

请于 AstrBot 的插件市场进行安装。AstrBot 将自动处理所需的 Python 库。

### 2. 安装 Playwright 浏览器驱动

为了正确提取网页内容，您需要手动安装 Playwright 的浏览器驱动。这是唯一需要手动安装的依赖。

- **对于本地或虚拟机安装**:
  ```bash
  python -m playwright install --with-deps
  ```

- **对于 Docker 用户**:
  为了持久化驱动，请在运行的容器中执行安装，并映射一个卷。

  1.  **在容器中安装 (一次性操作)**:
      ```bash
      docker exec -it <your_container_name_or_id> python -m playwright install --with-deps
      ```
  2.  **映射卷以持久化playwright安装**:
      在您的 `docker-compose.yml` 或 `docker run` 命令中，添加一个卷映射：
      ```yaml
      services:
        your_service:
          volumes:
            - ./playwright_cache:/root/.cache/ms-playwright
      ```

## ⚙️ 配置

在 AstrBot 的插件配置中，您可以为此插件添加特定配置：

```yaml
star_config:
  url_2_knowledge_base:
    debug_mode: false  # 设为 true 以在 pipeline/debug 目录下保存中间文件
    summarization_chunk_threshold: 10 # 当文本块数量超过此阈值时，启用聚类总结
    summarize_max_rpm: 20 # LLM 总结任务的每分钟最大请求数
    repair_max_rpm: 60 # LLM 修复任务的每分钟最大请求数
```

## 🔌 API 使用

插件注册了两个 API 端点，可通过 AstrBot 的 Web API 访问。

### 1. 提交 URL 处理任务

- **URL**: `/url_2_kb/add`
- **Method**: `POST`
- **Body (JSON)**:
  ```json
  {
    "url": "https://your-target-url.com",
    "use_llm_repair": true,
    "use_clustering_summary": true,
    "repair_llm_provider_id": "your-fast-llm-provider-id",
    "summarize_llm_provider_id": "your-powerful-llm-provider-id",
    "embedding_provider_id": "your-embedding-provider-id",
    "chunk_size": 300,
    "chunk_overlap": 50
  }
  ```
  - **`url`** (必需): 目标页面的 URL。
  - **`use_llm_repair`** (可选, 默认 `false`): 是否启用 LLM 文本修复。
  - **`use_clustering_summary`** (可选, 默认 `true`): 是否启用聚类总结。
  - **`repair_llm_provider_id`** (可选): 用于文本修复的 LLM Provider ID。
  - **`summarize_llm_provider_id`** (可选): 用于总结的 LLM Provider ID。
  - **`embedding_provider_id`** (可选): 用于文本嵌入的 Embedding Provider ID。
  - **`chunk_size`** (可选, 默认 `300`): 文本分块的大小。
  - **`chunk_overlap`** (可选, 默认 `50`): 文本分块的重叠部分大小。

- **成功响应 (202 Accepted)**:
  ```json
  {
    "status": "accepted",
    "message": "Task accepted for processing.",
    "task_id": "a-unique-task-id"
  }
  ```

### 2. 查询任务状态

- **URL**: `/url_2_kb/status`
- **Method**: `POST`
- **Body (JSON)**:
  ```json
  {
    "task_id": "the-task-id-you-received"
  }
  ```

- **响应**:
  - **处理中**:
    ```json
    {
      "task_id": "...",
      "status": "processing",
      "url": "https://..."
    }
    ```
  - **完成**:
    ```json
    {
      "task_id": "...",
      "status": "completed",
      "url": "https://...",
      "result": {
        "title": "页面标题",
        "content": "处理后的知识库内容...",
        "clusters": [
          {
            "cluster_id": 0,
            "summary": "类别 0 的摘要...",
            "docs": ["文本块1", "文本块2"]
          }
        ]
      }
    }
    ```
  - **失败**:
    ```json
    {
      "task_id": "...",
      "status": "failed",
      "url": "https://...",
      "error": "错误信息..."
    }
    ```

## 🤝 贡献

欢迎提交 Pull Request 或开启 Issue 来为这个项目做出贡献。

## 📄 开源许可

该项目基于 [AGPL-3.0 License](LICENSE) 开源。
