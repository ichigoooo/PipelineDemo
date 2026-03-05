"""
Layer 1: 元数据过滤
使用 ffprobe 检查视频基本属性：分辨率、时长、编码格式。
"""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VideoMeta:
    path: Path
    width: int
    height: int
    codec: str
    duration: float

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def is_landscape(self) -> bool:
        return self.width >= self.height


def probe_video(video_path: Path) -> Optional[VideoMeta]:
    """用 ffprobe 提取视频元数据，失败返回 None。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})
        return VideoMeta(
            path=video_path,
            width=int(stream.get("width", 0)),
            height=int(stream.get("height", 0)),
            codec=stream.get("codec_name", "unknown"),
            duration=float(fmt.get("duration", 0)),
        )
    except Exception as e:
        print(f"  [L1] ffprobe 失败 {video_path.name}: {e}")
        return None


def check_metadata(meta: VideoMeta, min_duration: float = 5.0, max_duration: float = 3600.0) -> tuple[bool, str]:
    """
    检查元数据是否通过 Layer 1 过滤。
    返回 (pass: bool, reason: str)
    """
    if meta.short_side < 720:
        return False, f"分辨率不足: 短边 {meta.short_side}px < 720px"
    if meta.duration < min_duration:
        return False, f"时长过短: {meta.duration:.1f}s < {min_duration}s"
    if meta.duration > max_duration:
        return False, f"时长过长: {meta.duration:.1f}s > {max_duration}s"
    if meta.codec not in ("h264", "hevc", "h265", "av1", "vp9"):
        return False, f"不常见编码: {meta.codec}（建议转码）"
    return True, "ok"
