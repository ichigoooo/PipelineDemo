"""
Layer 4: 片段级预筛 + 双人裁剪
- 单人片段：画质检测（模糊/亮度）后直接放行
- 双人片段（≥1080p）：FFmpeg crop+lanczos 放大，拆出两个单人片段
"""
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# 画质阈值
BLUR_THRESHOLD   = 50.0   # Laplacian 方差，低于此值视为模糊
DARK_THRESHOLD   = 30.0   # 灰度均值，低于此视为过暗
BRIGHT_THRESHOLD = 225.0  # 灰度均值，高于此视为过曝

# 裁剪参数
BBOX_PADDING_W  = 0.20   # 水平方向 padding（相对人体宽度）
BBOX_PADDING_H  = 0.10   # 垂直方向 padding（相对人体高度）
MAX_UPSCALE     = 1.5    # 最大允许放大倍率
MIN_SRC_HEIGHT  = 1080   # 触发双人裁剪的最小源视频高度（短边）

# 全身可见性检查（Pose 关键点）
# COCO 17关键点：0=鼻 1=左眼 2=右眼 3=左耳 4=右耳
#   5=左肩 6=右肩 ... 15=左踝 16=右踝
HEAD_KP_INDICES  = [0, 1, 2]    # 鼻、左眼、右眼（头部）
FEET_KP_INDICES  = [15, 16]     # 左踝、右踝（脚部）
KP_CONF_THRESHOLD = 0.3         # 关键点置信度阈值

# 人体高度占比要求（Blueprint 第6条）
HEIGHT_RATIO_LANDSCAPE = 0.50   # 横屏：人高 > 画面高度 1/2
HEIGHT_RATIO_PORTRAIT  = 0.33   # 竖屏：人高 > 画面高度 1/3


def extract_frames(video_path: Path, positions: list[float]) -> list[np.ndarray]:
    """
    按相对位置列表抽取帧，positions 为 0~1 的比例值。
    返回成功读取的帧列表（BGR numpy 数组）。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    frames = []
    for pos in positions:
        idx = int(total * pos)
        idx = max(0, min(total - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def extract_middle_frame(video_path: Path) -> Optional[np.ndarray]:
    """提取视频中间帧（兼容旧调用）。"""
    frames = extract_frames(video_path, [0.5])
    return frames[0] if frames else None


def full_body_check(keypoints, bbox_xyxy: list[float], frame_shape: tuple) -> tuple[bool, str]:
    """
    检查人体是否完整可见（需要 Pose 模型输出的关键点）。
    条件：
      1. 头部关键点（鼻/眼）至少 1 个置信度 > 阈值且在画面内
      2. 脚踝关键点至少 1 个置信度 > 阈值且在画面内
      3. 人体 bbox 高度占画面高度比例满足要求
    """
    h, w = frame_shape[:2]

    kp_xy   = keypoints.xy[0]    # (17, 2) 像素坐标
    kp_conf = keypoints.conf[0]  # (17,)   置信度

    def kp_visible(idx: int) -> bool:
        conf = float(kp_conf[idx])
        x, y = float(kp_xy[idx][0]), float(kp_xy[idx][1])
        return conf > KP_CONF_THRESHOLD and 0 <= x <= w and 0 <= y <= h

    head_visible = any(kp_visible(i) for i in HEAD_KP_INDICES)
    feet_visible = any(kp_visible(i) for i in FEET_KP_INDICES)

    x1, y1, x2, y2 = bbox_xyxy
    person_height = y2 - y1
    ratio = person_height / h
    min_ratio = HEIGHT_RATIO_LANDSCAPE if w >= h else HEIGHT_RATIO_PORTRAIT
    height_ok = ratio >= min_ratio

    if not head_visible:
        return False, "头部不可见（出画或被遮挡）"
    if not feet_visible:
        return False, "脚部不可见（出画或被遮挡）"
    if not height_ok:
        return False, f"人体占比不足 {ratio:.0%}（要求≥{min_ratio:.0%}）"
    return True, f"ok (head✓ feet✓ ratio={ratio:.0%})"


def quality_check(frame: np.ndarray) -> tuple[bool, str]:
    """
    基础画质检测：模糊 + 亮度。
    返回 (pass, reason)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = float(gray.mean())

    if blur_score < BLUR_THRESHOLD:
        return False, f"模糊 (Laplacian={blur_score:.1f})"
    if mean_brightness < DARK_THRESHOLD:
        return False, f"过暗 (mean={mean_brightness:.1f})"
    if mean_brightness > BRIGHT_THRESHOLD:
        return False, f"过曝 (mean={mean_brightness:.1f})"
    return True, f"ok (blur={blur_score:.1f}, bright={mean_brightness:.1f})"


def _calc_crop_box(box_xyxy: list[float], frame_shape: tuple, pad_w: float, pad_h: float) -> tuple[int, int, int, int]:
    """从 YOLO bbox 计算带 padding 的裁剪框（clamp 到帧边界）。"""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box_xyxy
    pw = (x2 - x1) * pad_w
    ph = (y2 - y1) * pad_h
    cx1 = max(0, int(x1 - pw))
    cy1 = max(0, int(y1 - ph))
    cx2 = min(w, int(x2 + pw))
    cy2 = min(h, int(y2 + ph))
    return cx1, cy1, cx2, cy2


def _calc_scale(crop_w: int, crop_h: int) -> float:
    """计算将裁剪区域放大到 ≥720p 所需的最小倍率。"""
    if crop_w >= crop_h:  # 横屏裁剪
        scale = max(1280 / crop_w, 720 / crop_h, 1.0)
    else:                 # 竖屏裁剪
        scale = max(720 / crop_w, 1280 / crop_h, 1.0)
    return scale


def try_crop_dual_person(
    clip_path: Path,
    persons: list,          # YOLO boxes，长度必须 == 2
    frame_shape: tuple,
    output_dir: Path,
) -> list[Path]:
    """
    双人裁剪：对 2 人分别 crop+scale，返回成功生成的片段路径列表（0-2 个）。
    """
    results = []
    for i, person in enumerate(persons):
        box = person.xyxy[0].tolist()
        cx1, cy1, cx2, cy2 = _calc_crop_box(box, frame_shape, BBOX_PADDING_W, BBOX_PADDING_H)
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        if crop_w <= 0 or crop_h <= 0:
            continue

        scale = _calc_scale(crop_w, crop_h)
        if scale > MAX_UPSCALE:
            continue  # 放大倍率过大，放弃

        out_w = (int(crop_w * scale) // 2) * 2
        out_h = (int(crop_h * scale) // 2) * 2

        stem = clip_path.stem
        out_path = output_dir / f"{stem}_crop{i}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-vf", f"crop={crop_w}:{crop_h}:{cx1}:{cy1},scale={out_w}:{out_h}:flags=lanczos",
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "copy",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            results.append(out_path)

    return results


# clip_prescreen 已移除：L4 逻辑已合并至 scan_and_segment.py（密集抽帧方案）
