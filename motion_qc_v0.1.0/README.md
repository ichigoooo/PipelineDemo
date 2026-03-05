# Motion QC

基于视觉语言模型（VLM）的**视频帧质量检测工具**。
从视频生成缩略图 → 调用多模态模型分析 → 输出结构化质检结果。

---

## 功能特点

* 自动生成视频缩略图
* 调用多模态大模型进行画面分析
* 自动提取 JSON 结果
* 使用 Pydantic 结构化解析
* 模块解耦，易于扩展

---

## 核心流程

```text
视频 → 缩略图 → 图像编码 → 构建Prompt → LLM分析 → JSON解析 → 结构化结果
```

---

## 快速使用

```python
from motion_qc.media.thumb import gen_video_thumb
from motion_qc.vlm.chatbot import ChatBot
from motion_qc.vlm.prompt import build_messages
from motion_qc.vlm.config import llm_config
from motion_qc.types import FrameCheckResult

chatbot = ChatBot(llm_config.get("general", {}), silent=True)

# 生成缩略图
img_path = gen_video_thumb(
    video_path="input.mp4",
    thumbnail_path="thumb.jpg",
    width=720,
)

# 构建请求
img_b64 = chatbot.encode_image(img_path.as_posix())
msgs = build_messages(img_b64)

# 获取结果
resp = chatbot.chat_with_messages(msgs)
resp_json = chatbot.extract_json(resp)

result = FrameCheckResult.from_json(resp_json)
print(result)
```

---

## 目录结构

```
src/motionx_qa
├── media/      # 视频缩略图生成
├── vlm/        # 模型调用与 prompt
└── types.py    # 结构化结果定义
```

---

## 适用场景

* 视频自动审核
* 画面质量检测
* 批量视频数据分析

轻量、模块化，适合集成到自动化流水线中。


