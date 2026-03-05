
from pathlib import Path
from motion_qc.media.executor import execute_ffmpeg_command

def gen_video_thumb(
    video_path: str | Path,
    thumbnail_path: str | Path,
    width: int = 720,
    skip_exist: bool = False,
    verbose: bool = False
) -> Path | None:
    """
    使用 FFmpeg 生成视频缩略图 (假设视频片段超过1s, 取第1s的视频帧)
    输入 video_path，输出 thumbnail_path（已覆盖旧图）

    Args:
      video_path: 源视频文件
      thumbnail_path: 缩略图输出完整路径
      width: 缩略图宽度（按比例缩放）
      verbose: 是否打印 ffmpeg 输出信息

    Returns:
      thumbnail_path 对应的 Path 对象
    """
    video = Path(video_path)
    thumb = Path(thumbnail_path)

    if skip_exist and thumb.exists():
        return thumb  # 如果缩略图已存在，则直接返回
    
    thumb.parent.mkdir(parents=True, exist_ok=True)

    # 构建命令
    cmd = [
        "ffmpeg", "-y",
        "-ss", "1",  # 跳转到 1 秒
        "-skip_frame", "nokey",   # 只解码关键帧
        "-i", str(video),
        "-vf", f"scale={width}:-1:flags=fast_bilinear",
        "-update", "1",
        "-frames:v", "1",
        "-c:v", "mjpeg",     # 显式指定 jpeg 编码
        "-q:v", "8",         # 控制压缩质量
        "-threads", "1",      # 禁用多线程, 高并发更稳定
        str(thumb),
    ]

    # 然后调用
    ret = execute_ffmpeg_command(cmd, verbose)

    if ret:
        return thumb
    else:
        print(f"生成 {video.name} 缩略图失败")
        return None

if __name__ == "__main__":
    
  src = "/data/source/yubobang/260224_2103/bilibili_BV1PifABaE8R_1080x1920_993s_seg_5_20s.mp4"
  jpg = "/data/thumbnail/yubobang/260224_2103/bilibili_BV1PifABaE8R_1080x1920_993s_seg_5_20s.jpg"
  
  gen_video_thumb(src, jpg)