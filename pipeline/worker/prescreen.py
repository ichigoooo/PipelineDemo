"""
Layer 2: 视频级 YOLO 预筛
从视频均匀抽取 5 帧，YOLO 检测人数，三分类：SINGLE / DUAL / REJECT。
"""
from enum import Enum
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import os

import cv2
import numpy as np


class VideoVerdict(Enum):
    SINGLE = "single"   # ≥3/5 帧单人 → 正常流程
    DUAL   = "dual"     # ≥3/5 帧双人 → 标记裁剪候选
    REJECT = "reject"   # 其他 → 丢弃


def extract_uniform_frames(video_path: Path, n: int = 5) -> list[np.ndarray]:
    """用 ffmpeg 均匀抽取 n 帧，返回 BGR numpy 数组列表。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    if total <= 0:
        cap.release()
        return []

    indices = [int(total * i / (n + 1)) for i in range(1, n + 1)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def video_prescreen(video_path: Path, yolo_model) -> VideoVerdict:
    """
    均匀抽取 5 帧，YOLO 检测每帧人数，投票三分类。
    yolo_model: ultralytics YOLO 实例（yolov8n-pose 或 yolov8n）
    """
    frames = extract_uniform_frames(video_path, n=5)
    if not frames:
        return VideoVerdict.REJECT

    counts = {1: 0, 2: 0}
    for frame in frames:
        results = yolo_model(frame, verbose=False)
        persons = [
            b for b in results[0].boxes
            if int(b.cls[0]) == 0 and float(b.conf[0]) > 0.5
        ]
        n = len(persons)
        if n in counts:
            counts[n] += 1

    if counts[1] >= 3:
        return VideoVerdict.SINGLE
    if counts[2] >= 3:
        return VideoVerdict.DUAL
    return VideoVerdict.REJECT
