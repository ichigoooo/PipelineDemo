
from motion_qc.media.thumb import gen_video_thumb

from motion_qc.vlm.chatbot import ChatBot
from motion_qc.vlm.prompt import build_messages
from motion_qc.vlm.config import llm_config

from motion_qc.types import FrameCheckResult

from rich import print as rprint


chatbot = ChatBot(llm_config.get("general", {}), silent=True)


if __name__ == "__main__":
    # 定义输入的视频路径 和 要生成的缩略图路径
    video_path = r"tmp\src\BV1F44y1X7gE_scenes\BV1F44y1X7gE_006_013s.mp4"
    thumbnail_path = r"thumb\src\BV1F44y1X7gE_scenes\BV1F44y1X7gE_006_013s.jpg"

    # 生成缩略图
    img_path = gen_video_thumb(
        video_path=video_path,
        thumbnail_path=thumbnail_path,
        width=720,
        skip_exist=False,
        verbose=False,
    )

    if img_path is None:
        raise ValueError("未能生成缩略图")

    # 构建消息
    img_b64 = chatbot.encode_image(img_path.as_posix())
    msgs = build_messages(img_b64)

    # 发送消息
    resp_str = chatbot.chat_with_messages(msgs)
    rprint(resp_str) 

    if resp_str is None:
        raise ValueError("未能获取到 LLM 响应")
    
    # 从文本中提取 JSON 部分 (可选)
    resp_json = chatbot.extract_json(resp_str)
    rprint(resp_json)
    if resp_json is None:
        raise ValueError("未能从LLM 响应文本提取到 JSON")

    # 使用 Pydantic 结构化解析 LLM 响应
    check_result = FrameCheckResult.from_json(resp_json)
    rprint(check_result)

