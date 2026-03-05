"""
视频筛选主 Pipeline（Layer 1-5）
用法:
  python run_pipeline.py --input <视频目录或单个视频> [--output <输出目录>] [--skip-vlm]

Layer 0（批次接入）在本脚本中以简化方式集成（扫描目录）。
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── 依赖检查 ────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] 请先安装: pip install ultralytics")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] 请先安装: pip install opencv-python-headless")
    sys.exit(1)

# ── 本地模块 ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from worker.metadata_filter import probe_video, check_metadata
from worker.prescreen import video_prescreen, VideoVerdict
from worker.scan_and_segment import scan_and_segment


# ═══════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VideoResult:
    video_name: str
    verdict_l1: str = "pending"    # pass / fail / error
    verdict_l2: str = "pending"    # single / dual / reject / error
    n_segments: int = 0
    n_clips_pass_l4: int = 0
    n_clips_pass_l5: int = 0
    clips_pass: list[str] = field(default_factory=list)
    clips_fail_l4: list[str] = field(default_factory=list)
    vlm_results: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    def summary(self) -> str:
        return (
            f"{self.video_name}: "
            f"L1={self.verdict_l1} L2={self.verdict_l2} "
            f"windows={self.n_segments} "
            f"clips={self.n_clips_pass_l4} "
            f"L5_pass={self.n_clips_pass_l5}"
        )


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def process_video(
    video_path: Path,
    output_dir: Path,
    yolo_model,
    chatbot=None,
    skip_vlm: bool = False,
) -> VideoResult:
    result = VideoResult(video_name=video_path.name)
    video_id = video_path.stem

    # ── Layer 1: 元数据过滤 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[视频] {video_path.name}")
    meta = probe_video(video_path)
    if meta is None:
        result.verdict_l1 = "error"
        result.error = "ffprobe 失败"
        return result

    passed, reason = check_metadata(meta)
    print(f"  [L1] {meta.width}x{meta.height} {meta.codec} {meta.duration:.1f}s → {'✓' if passed else '✗'} {reason}")
    if not passed:
        result.verdict_l1 = "fail"
        result.error = reason
        return result
    result.verdict_l1 = "pass"

    # ── Layer 2: 视频级 YOLO 预筛 ────────────────────────────────────
    verdict = video_prescreen(video_path, yolo_model)
    result.verdict_l2 = verdict.value
    print(f"  [L2] YOLO 三分类 → {verdict.value.upper()}")
    if verdict == VideoVerdict.REJECT:
        return result

    # ── Layer 3+4: 密集扫描 + 连续窗口提取 ──────────────────────────
    clips_dir = output_dir / "clips" / video_id
    passed_clips, scan_stats = scan_and_segment(video_path, clips_dir, video_id, yolo_model)
    result.n_segments = scan_stats["n_windows"]
    result.n_clips_pass_l4 = len(passed_clips)
    result.clips_pass = [str(p) for p in passed_clips]
    cam_info = ""
    if scan_stats.get("n_cam_rejected", 0) > 0:
        cam_info = f" (摄像机移动淘汰 {scan_stats['n_cam_rejected']})"
    print(
        f"  [L3+L4] 扫描 {scan_stats['n_frames_scanned']} 帧 → "
        f"{scan_stats['n_frames_valid']} 合格帧 → "
        f"{scan_stats['n_windows_raw']} 窗口{cam_info} → "
        f"{scan_stats['n_windows']} 固定机位窗口 → "
        f"{len(passed_clips)} 片段"
    )

    # ── Layer 5: VLM 质检 ────────────────────────────────────────────
    if skip_vlm or chatbot is None:
        print(f"  [L5] 跳过 VLM 质检（--skip-vlm）")
        result.n_clips_pass_l5 = result.n_clips_pass_l4
        return result

    thumb_dir = output_dir / "thumbs" / video_id
    vlm_pass_clips = []

    for clip in passed_clips:
        from worker.qc_runner import vlm_check
        qc_result = vlm_check(clip, chatbot, thumb_dir)
        if qc_result is None:
            print(f"    [L5] ✗ {clip.name} (VLM 调用失败)")
            continue

        result.vlm_results.append({
            "clip": clip.name,
            "passed": qc_result.passed,
            "problems": [p.value for p in qc_result.problems] if qc_result.problems else None,
            "description": qc_result.description,
            "comment": qc_result.comment,
        })

        icon = "✓" if qc_result.passed == "pass" else "✗"
        prob_str = ",".join(p.value for p in qc_result.problems) if qc_result.problems else "-"
        print(f"    [L5] {icon} {clip.name} | {qc_result.description} | {prob_str}")

        if qc_result.passed == "pass":
            vlm_pass_clips.append(str(clip))

    result.n_clips_pass_l5 = len(vlm_pass_clips)
    result.clips_pass = vlm_pass_clips
    print(f"  [L5] {len(vlm_pass_clips)}/{result.n_clips_pass_l4} 片段 VLM 通过")

    return result


# ═══════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════

def collect_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("**/*.mp4")) + sorted(input_path.glob("**/*.MP4"))


def main():
    parser = argparse.ArgumentParser(description="视频筛选 Pipeline (Layer 1-5)")
    parser.add_argument("--input",  "-i", required=True, help="输入视频目录或单个视频文件")
    parser.add_argument("--output", "-o", default="./output", help="输出目录（默认 ./output）")
    parser.add_argument("--skip-vlm", action="store_true", help="跳过 Layer 5 VLM 质检（节省 API 费用）")
    parser.add_argument("--yolo-model", default="yolov8n-pose.pt", help="YOLO 模型路径（默认 yolov8n-pose.pt）")
    parser.add_argument("--vlm-backend", default="general", choices=["general", "ollama", "ucloud"],
                        help="VLM 后端（默认 general=qwen3.5-flash）")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(input_path)
    if not videos:
        print(f"[ERROR] 未找到视频文件: {input_path}")
        sys.exit(1)
    print(f"[INFO] 发现 {len(videos)} 个视频，输出目录: {output_dir}")

    # 加载 YOLO
    print(f"[INFO] 加载 YOLO 模型: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)

    # 加载 VLM（可选）
    chatbot = None
    if not args.skip_vlm:
        try:
            from worker.qc_runner import make_chatbot
            chatbot = make_chatbot(args.vlm_backend)
            print(f"[INFO] VLM 后端: {args.vlm_backend}")
        except Exception as e:
            print(f"[WARN] VLM 初始化失败，将跳过 L5: {e}")
            args.skip_vlm = True

    # 处理每个视频
    all_results = []
    t0 = time.time()

    for video_path in videos:
        r = process_video(video_path, output_dir, yolo, chatbot, skip_vlm=args.skip_vlm)
        all_results.append(r)
        print(f"  → {r.summary()}")

    elapsed = time.time() - t0

    # 汇总报告
    print(f"\n{'='*60}")
    print(f"处理完成，耗时 {elapsed:.1f}s")
    print(f"{'='*60}")
    total_l4 = sum(r.n_clips_pass_l4 for r in all_results)
    total_l5 = sum(r.n_clips_pass_l5 for r in all_results)
    total_windows = sum(r.n_segments for r in all_results)
    rejected = sum(1 for r in all_results if r.verdict_l2 == "reject")
    print(f"视频总数    : {len(all_results)}")
    print(f"L2 淘汰     : {rejected}")
    print(f"有效窗口    : {total_windows}")
    print(f"输出片段    : {total_l4}")
    if not args.skip_vlm:
        print(f"L5 VLM 通过 : {total_l5}")

    # 保存 JSON 报告
    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
