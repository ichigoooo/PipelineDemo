import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
UCLOUD_BASE_URL = "http://127.0.0.1:8888/v1"

# 检查必要环境变量是否存在
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

general_model = "qwen3.5-flash"
# general_model = "qwen3.5-plus"

llm_config: dict = {
    "general": {
        "model": general_model,
        "system_prompt": " ",
        "api_key": DASHSCOPE_API_KEY,
        "base_url": ALIYUN_BASE_URL,
    },
    "ollama": {
        "model": "qwen3.5:35b-a3b",
        "api_key": "any_string",
        "base_url": OLLAMA_BASE_URL,
    },
    "ucloud": {
        "model": "Qwen/Qwen3.5-35B-A3B-FP8",
        "api_key": "any_string",
        "base_url": UCLOUD_BASE_URL,
    },
}
