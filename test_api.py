import requests
import json
import time

# --- 配置 ---
ASTRBOT_HOST = "127.0.0.1"
ASTRBOT_PORT = 6185
TARGET_URL = "https://baike.baidu.com/item/Python/407313"
POLLING_INTERVAL = 5  # 轮询间隔（秒）

# --- API 地址 ---
base_url = f"http://{ASTRBOT_HOST}:{ASTRBOT_PORT}/api/plug/url_2_kb"
add_url = f"{base_url}/add"
status_url = f"{base_url}/status"

# --- 认证 Token ---
AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImFzdHJib3QiLCJleHAiOjE3NTMyMzMxMTF9.AwZhgvbVTw4aHIJm4tEYyoxqFe_X5nL5XWgILVBkKqo"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH_TOKEN}"
}

def start_task():
    """发送请求以启动任务并获取 task_id"""
    payload = {
        "url": TARGET_URL,
        "use_llm_repair": True,
        "use_clustering_summary": True,
        "repair_llm_provider_id": "oneapi_lite",
        "summarize_llm_provider_id": "oneapi",
        "embedding_provider_id": "oneapi_bgem3",
        "chunk_size": 400,
        "chunk_overlap": 100
    }
    print(f"🚀 1. 向 {add_url} 发送请求以启动任务...")
    print(f"   请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    response = requests.post(add_url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    
    data = response.json()
    print(f"✅ 任务已成功创建, Task ID: {data['task_id']}")
    return data['task_id']

def poll_status(task_id):
    """轮询任务状态直到完成或失败"""
    print(f"\n🔄 2. 开始轮询任务状态 (每 {POLLING_INTERVAL} 秒一次)...")
    
    while True:
        print(f"   向 {status_url} 发送 POST 请求查询任务状态...")
        payload = {"task_id": task_id}
        response = requests.post(status_url, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        status = data.get("status")
        
        print(f"   响应内容: {json.dumps(data, indent=2, ensure_ascii=False)}")

        if status == "completed":
            print("\n🎉 任务成功完成!")
            break
        elif status == "failed":
            print("\n❌ 任务失败!")
            break
        else:
            print(f"   任务仍在处理中... 等待 {POLLING_INTERVAL} 秒后重试...")
            time.sleep(POLLING_INTERVAL)

if __name__ == "__main__":
    try:
        task_id = start_task()
        poll_status(task_id)
    except requests.exceptions.RequestException as e:
        print(f"\n发生网络或 HTTP 错误: {e}")
        print("请确认 AstrBot 是否正在运行，并且地址、端口和 Token 配置正确。")
    except Exception as e:
        print(f"\n发生未知错误: {e}")