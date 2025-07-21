import requests
import json

# --- 配置 ---
# AstrBot 仪表盘的地址和端口
ASTRBOT_HOST = "127.0.0.1"
ASTRBOT_PORT = 6185
# 要处理的目标 URL
TARGET_URL = "https://zh.wikipedia.org/wiki/%E4%B8%8A%E6%B5%B7%E5%9B%9B%E4%B8%80%E4%B8%80%E5%8C%BB%E9%99%A2#%E5%8F%98%E6%80%A7%E6%89%8B%E6%9C%AF"

# 构建完整的 API 地址
api_url = f"http://{ASTRBOT_HOST}:{ASTRBOT_PORT}/api/plug/url_2_kb/add"

# 构建请求体
payload = {
    "url": TARGET_URL,
    "use_llm_repair": True,  # 是否使用 LLM 修复内容，默认为 False
    "use_clustering_summary": True  # 是否进行聚类总结，默认为 True
}

# --- 认证 Token ---
AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImFzdHJib3QiLCJleHAiOjE3NTMyMzMxMTF9.AwZhgvbVTw4aHIJm4tEYyoxqFe_X5nL5XWgILVBkKqo"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH_TOKEN}"
}

print(f"正在向 {api_url} 发送 POST 请求...")
print(f"请求体: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    # 检查响应状态码
    if response.status_code == 202:
        print("\n请求成功! 服务器已接收请求并开始在后台处理。")
        print("请检查 AstrBot 的控制台日志以查看处理进度。")
        print(f"响应内容: {response.json()}")
    else:
        print(f"\n请求失败! 状态码: {response.status_code}")
        try:
            # 尝试打印 JSON 格式的错误信息
            print(f"错误详情: {response.json()}")
        except json.JSONDecodeError:
            # 如果响应不是 JSON，则直接打印文本内容
            print(f"响应内容: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"\n发生网络错误: {e}")
    print("请确认 AstrBot 是否正在运行，并且地址和端口配置正确。")
