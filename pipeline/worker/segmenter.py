"""
Layer 3: PySceneDetect 场景切割 + 时长过滤
检测剪辑跳切点，将视频分割为连续无跳切片段，长片段均匀切割至 10-20s。
"""
import subprocess
from pathlib import Path
from typing import Optional

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


# 片段时长约束
MIN_CLIP_DURATION = 5.0   # 秒
MAX_CLIP_DURATION = 20.0  # 秒
TARGET_CLIP_DURATION = 15.0  # 长片段分割目标长度


def detect_scenes(video_path: Path, threshold: float = 27.0) -> list[tuple[float, float]]:
    """
    使用 PySceneDetect ContentDetector 检测场景边界。
    返回 [(start_sec, end_sec), ...] 的连续片段列表。
    """
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video, show_progress=False)
    scene_list = manager.get_scene_list()

    if not scene_list:
        # 无切换点：整段视频视为一个场景
        cap_dur = video.duration.get_seconds()
        return [(0.0, cap_dur)]

    return [
        (scene[0].get_seconds(), scene[1].get_seconds())
        for scene in scene_list
    ]


def split_long_segment(start: float, end: float) -> list[tuple[float, float]]:
    """将超过 MAX_CLIP_DURATION 的片段均匀分割为若干子片段。"""
    import math
    duration = end - start
    if duration <= MAX_CLIP_DURATION:
        return [(start, end)]

    # ceil 确保 20~30s 的片段被切成 2 段，而不是 int() 取整后变成 1 段
    n = math.ceil(duration / TARGET_CLIP_DURATION)
    step = duration / n
    segments = []
    for i in range(n):
        s = start + i * step
        e = s + step
        segments.append((round(s, 3), round(e, 3)))
    return segments


def get_clip_segments(video_path: Path, threshold: float = 27.0) -> list[tuple[float, float]]:
    """
    完整 Layer 3 流程：检测场景 → 分割长片段 → 过滤短片段。
    返回满足时长要求的 (start, end) 列表。
    """
    raw_scenes = detect_scenes(video_path, threshold)
    segments = []
    for start, end in raw_scenes:
        for sub_start, sub_end in split_long_segment(start, end):
            if (sub_end - sub_start) >= MIN_CLIP_DURATION:
                segments.append((sub_start, sub_end))
    return segments


MAX_ACTUAL_DURATION = 20.0  # -c copy 关键帧对齐允许的最大实际时长容差（对齐交付规范 5-20s）


def _get_duration(path: Path) -> float:
    """用 ffprobe 获取文件实际时长。"""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def cut_clip(video_path: Path, start: float, duration: float, output_path: Path) -> bool:
    """
    用 FFmpeg 切割片段（-c copy 零损耗，关键帧对齐）。
    切割后校验实际时长，超过 MAX_ACTUAL_DURATION 则删除并返回 False。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    # 校验实际时长，-c copy 关键帧对齐可能导致轻微超时
    actual = _get_duration(output_path)
    if actual > MAX_ACTUAL_DURATION or actual < MIN_CLIP_DURATION:
        output_path.unlink(missing_ok=True)
        return False
    return True


def segment_video(
    video_path: Path,
    output_dir: Path,
    video_id: str,
    threshold: float = 27.0,
) -> list[Path]:
    """
    完整切割流程：检测 + 切割 + 返回片段路径列表。
    """
    segments = get_clip_segments(video_path, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = []

    for i, (start, end) in enumerate(segments):
        duration = end - start
        clip_name = f"{video_id}_{i:03d}.mp4"
        out_path = output_dir / clip_name
        if cut_clip(video_path, start, duration, out_path):
            clip_paths.append(out_path)
        else:
            print(f"  [L3] 切割失败: {clip_name}")

    return clip_paths
