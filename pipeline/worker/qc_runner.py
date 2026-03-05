"""
Layer 5: VLM 质检封装
调用 motion_qc 库对片段进行单帧 VLM 分析，返回 FrameCheckResult。
"""
import sys
from pathlib import Path
from typing import Optional

# 将 motion_qc 源码目录加入 sys.path
_MOTION_QC_SRC = Path(__file__).parent.parent.parent / "motion_qc_v0.1.0" / "src"
if str(_MOTION_QC_SRC) not in sys.path:
    sys.path.insert(0, str(_MOTION_QC_SRC))

from motion_qc.media.thumb import gen_video_thumb
from motion_qc.vlm.chatbot import ChatBot
from motion_qc.vlm.prompt import build_messages
from motion_qc.vlm.config import llm_config
from motion_qc.types import FrameCheckResult


def make_chatbot(backend: str = "general") -> ChatBot:
    """创建 ChatBot 实例，backend 可选 general / ollama / ucloud。"""
    return ChatBot(llm_config[backend], silent=True)


def vlm_check(
    clip_path: Path,
    chatbot: ChatBot,
    thumb_dir: Path,
) -> Optional[FrameCheckResult]:
    """
    对单个片段进行 VLM 质检。
    1. 提取第 1 秒关键帧 → 720px 宽 JPEG
    2. BASE64 编码 → 构建 prompt → 调用 API
    3. 解析返回 FrameCheckResult
    """
    thumb_path = thumb_dir / f"{clip_path.stem}_thumb.jpg"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    generated = gen_video_thumb(clip_path, thumb_path, width=720)
    if generated is None:
        print(f"  [L5] 缩略图生成失败: {clip_path.name}")
        return None

    try:
        img_b64 = chatbot.encode_image(generated.as_posix())
        msgs = build_messages(img_b64)
        resp = chatbot.chat_with_messages(msgs)
        json_str = chatbot.extract_json(resp)
        if json_str is None:
            print(f"  [L5] JSON 提取失败: {clip_path.name}")
            return None
        return FrameCheckResult.from_json(json_str)
    except Exception as e:
        print(f"  [L5] VLM 调用异常 {clip_path.name}: {e}")
        return None
