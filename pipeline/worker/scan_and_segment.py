"""
L3+L4 合并：密集抽帧 → 连续有效窗口提取

取代原来的 segmenter.py（L3）和 clip_filter.py clip_prescreen（L4）。
对每个视频做密集逐帧扫描，在时间轴上标记每帧是否合格，
找出所有连续合格区间（≥5s），均匀切成 5-15s 的片段输出。

流程:
  视频 → PySceneDetect 场景边界（防止切割跨硬切）
       → 对每个场景内，每 0.5s 抽一帧
       → 每帧评估：画质 + 单人 + 全身可见
       → 构建布尔时间轴 [T,T,T,F,F,T,T,T,T,T...]
       → 场景边界处插入 False 屏障，确保窗口不跨硬切
       → 找所有连续 True 区间 ≥ 5s
       → 区间 > 15s 均匀切成多个 5-15s 子片段
       → 逐窗口检测摄像机运动（背景特征点跟踪），过滤跟踪镜头
       → FFmpeg 切割 → 验证片段开头 → 如有脏帧则重编码降级
"""
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from .clip_filter import quality_check, full_body_check
from .segmenter import cut_clip


# ── 关键参数 ─────────────────────────────────────────────────────────────────
SAMPLE_INTERVAL    = 0.5    # 秒，抽帧间隔
MIN_VALID_DURATION = 5.0    # 秒，最短有效窗口
MAX_VALID_DURATION = 15.0   # 秒，最长有效窗口（超过则均匀切割）
MAX_ACTUAL_DURATION = 20.0  # 秒，-c copy / re-encode 后实际时长上限（对齐交付规范）
VERIFY_DURATION    = 3.0    # 秒，切割后验证片段开头的长度

# YOLO 检测参数
YOLO_PERSON_CONF = 0.5      # 人体检测置信度阈值

# 摄像机运动检测参数
CAMERA_SAMPLE_GAP     = 0.5    # 秒，帧对间隔（与 SAMPLE_INTERVAL 一致）
CAMERA_CHECK_INTERVAL = 1.0    # 秒，窗口内每隔多久取一个帧对检测
CAMERA_TRANS_THRESH   = 2.0    # 像素，背景特征点中位平移量阈值（0.5s 间隔）
CAMERA_MIN_FEATURES   = 15     # 最少背景特征点数量（低于则跳过该帧对）
CAMERA_BBOX_PAD       = 0.25   # 人体 bbox 扩展比例（遮掉运动区域）


# ═══════════════════════════════════════════════════════════════════
# Step 1：场景边界检测
# ═══════════════════════════════════════════════════════════════════

def detect_scene_boundaries(
    video_path: Path,
    threshold: float = 27.0,
) -> list[tuple[float, float]]:
    """
    用 PySceneDetect ContentDetector 检测场景边界。
    返回 [(start_sec, end_sec), ...]，无切换点时整段作为单一场景。
    """
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video, show_progress=False)
    scene_list = manager.get_scene_list()

    if not scene_list:
        return [(0.0, video.duration.get_seconds())]

    return [
        (scene[0].get_seconds(), scene[1].get_seconds())
        for scene in scene_list
    ]


# ═══════════════════════════════════════════════════════════════════
# Step 2：逐帧评估
# ═══════════════════════════════════════════════════════════════════

def evaluate_frame(frame: np.ndarray, yolo_model) -> tuple[bool, str]:
    """
    评估单帧是否合格。三个条件全部满足才算通过：
      1. 画质：Laplacian 方差 > 50，灰度均值 30~225
      2. 单人：YOLO Pose 检测到恰好 1 人（conf > 0.5）
      3. 全身可见：头部关键点（鼻/眼）+ 脚踝关键点均可见，人高占比 ≥ 50%（横屏）
    返回 (is_valid, reason)。
    """
    # ① 画质检测
    quality_ok, quality_reason = quality_check(frame)
    if not quality_ok:
        return False, quality_reason

    # ② YOLO Pose 人数检测
    results = yolo_model(frame, verbose=False)
    res = results[0]
    persons = [
        b for b in res.boxes
        if int(b.cls[0]) == 0 and float(b.conf[0]) > YOLO_PERSON_CONF
    ]
    n = len(persons)
    if n != 1:
        return False, f"人数={n}（需要恰好1人）"

    # ③ 全身可见性（关键点检查）
    kps = res.keypoints
    if kps is None or len(kps.xy) == 0:
        return False, "无关键点数据"

    body_ok, body_reason = full_body_check(kps, persons[0].xyxy[0].tolist(), frame.shape)
    if not body_ok:
        return False, body_reason

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════
# Step 3：构建时间轴
# ═══════════════════════════════════════════════════════════════════

def build_frame_timeline(
    video_path: Path,
    yolo_model,
    scenes: list[tuple[float, float]],
) -> list[tuple[float, bool, str]]:
    """
    按场景边界对视频做密集抽帧，返回 [(timestamp, is_valid, reason), ...]。
    每帧间隔 SAMPLE_INTERVAL 秒，严格在场景边界内采样（不跨硬切）。
    在相邻场景之间插入 False 屏障条目，确保 find_valid_windows 不会
    把跨硬切的帧合并为一个连续窗口。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    timeline: list[tuple[float, bool, str]] = []

    for i, (scene_start, scene_end) in enumerate(scenes):
        t = scene_start
        # 留 10% 采样间隔的容差，避免浮点边界问题
        boundary = scene_end - SAMPLE_INTERVAL * 0.1
        while t < boundary:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret:
                break
            valid, reason = evaluate_frame(frame, yolo_model)
            timeline.append((t, valid, reason))
            t += SAMPLE_INTERVAL

        # 在场景之间插入屏障，防止窗口跨硬切
        if i < len(scenes) - 1:
            timeline.append((scene_end, False, "scene_boundary"))

    cap.release()
    return timeline


# ═══════════════════════════════════════════════════════════════════
# Step 4：找连续有效窗口
# ═══════════════════════════════════════════════════════════════════

def _split_window(start: float, end: float) -> list[tuple[float, float]]:
    """将超过 MAX_VALID_DURATION 的窗口均匀切成若干子窗口。"""
    duration = end - start
    if duration <= MAX_VALID_DURATION:
        return [(round(start, 3), round(end, 3))]

    n = math.ceil(duration / MAX_VALID_DURATION)
    step = duration / n
    return [
        (round(start + i * step, 3), round(start + (i + 1) * step, 3))
        for i in range(n)
    ]


def find_valid_windows(
    timeline: list[tuple[float, bool, str]],
) -> list[tuple[float, float]]:
    """
    从布尔时间轴找出所有连续 True 区间（≥ MIN_VALID_DURATION），
    超过 MAX_VALID_DURATION 则均匀切割成子窗口。
    返回 [(start_sec, end_sec), ...]。

    窗口端点规则：
      - window_start = 第一个合格帧的时间戳
      - window_end   = 最后一个合格帧的时间戳 + SAMPLE_INTERVAL
        （包含最后一帧所对应的完整采样区间）
    """
    if not timeline:
        return []

    windows: list[tuple[float, float]] = []
    in_window = False
    window_start = 0.0
    last_valid_t = 0.0

    for t, valid, _reason in timeline:
        if valid and not in_window:
            in_window = True
            window_start = t
            last_valid_t = t
        elif valid and in_window:
            last_valid_t = t
        elif not valid and in_window:
            in_window = False
            window_end = last_valid_t + SAMPLE_INTERVAL
            if window_end - window_start >= MIN_VALID_DURATION:
                windows.extend(_split_window(window_start, window_end))

    # 处理末尾仍在窗口内的情况
    if in_window:
        window_end = last_valid_t + SAMPLE_INTERVAL
        if window_end - window_start >= MIN_VALID_DURATION:
            windows.extend(_split_window(window_start, window_end))

    return windows


# ═══════════════════════════════════════════════════════════════════
# Step 5：摄像机运动检测（固定机位过滤）
# ═══════════════════════════════════════════════════════════════════

def _make_person_mask(
    frame: np.ndarray, yolo_model, pad: float = CAMERA_BBOX_PAD,
) -> np.ndarray:
    """
    用 YOLO 检测人体，生成前景 mask（255=人体区域，0=背景）。
    bbox 向外扩展 pad 比例，覆盖肢体运动范围。
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    results = yolo_model(frame, verbose=False)
    for box in results[0].boxes:
        if int(box.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * pad))
        y1 = max(0, int(y1 - bh * pad))
        x2 = min(w, int(x2 + bw * pad))
        y2 = min(h, int(y2 + bh * pad))
        mask[y1:y2, x1:x2] = 255
    return mask


def check_camera_static(
    video_path: Path,
    window_start: float,
    window_end: float,
    yolo_model,
) -> tuple[bool, float]:
    """
    检测窗口内摄像机是否固定。

    方法：在窗口内每隔 CAMERA_CHECK_INTERVAL 取一对帧（间隔 CAMERA_SAMPLE_GAP），
    用 YOLO bbox 遮掉人体前景，在背景区域检测特征点并追踪光流，
    用 estimateAffinePartial2D（内置 RANSAC）估计全局运动，提取平移量。

    返回 (is_static, median_translation_px)。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return True, 0.0  # 无法打开则放行

    translations: list[float] = []
    t = window_start

    while t + CAMERA_SAMPLE_GAP <= window_end:
        # 读取帧对
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret1, frame1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, (t + CAMERA_SAMPLE_GAP) * 1000.0)
        ret2, frame2 = cap.read()

        if not ret1 or not ret2:
            t += CAMERA_CHECK_INTERVAL
            continue

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 生成人体前景 mask，取反得到背景 mask
        fg_mask = _make_person_mask(frame1, yolo_model)
        bg_mask = cv2.bitwise_not(fg_mask)

        # 在背景区域检测特征点
        features = cv2.goodFeaturesToTrack(
            gray1, maxCorners=300, qualityLevel=0.01,
            minDistance=20, mask=bg_mask,
        )

        if features is None or len(features) < CAMERA_MIN_FEATURES:
            t += CAMERA_CHECK_INTERVAL
            continue

        # 光流追踪到下一帧
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, features, None)
        good_prev = features[status.flatten() == 1]
        good_curr = tracked[status.flatten() == 1]

        if len(good_prev) < CAMERA_MIN_FEATURES:
            t += CAMERA_CHECK_INTERVAL
            continue

        # 估计仿射变换（RANSAC 自动过滤局部运动干扰如树叶、路人）
        A, _inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)
        if A is not None:
            tx, ty = A[0, 2], A[1, 2]
            translations.append(math.sqrt(tx * tx + ty * ty))

        t += CAMERA_CHECK_INTERVAL

    cap.release()

    if not translations:
        return True, 0.0  # 无数据则放行

    median_trans = float(np.median(translations))
    return median_trans < CAMERA_TRANS_THRESH, median_trans


def filter_windows_by_camera(
    video_path: Path,
    windows: list[tuple[float, float]],
    yolo_model,
    verbose: bool = True,
) -> list[tuple[float, float]]:
    """
    过滤掉摄像机非固定的窗口。
    返回仅保留固定机位的窗口列表。
    """
    static_windows: list[tuple[float, float]] = []
    for i, (start, end) in enumerate(windows):
        is_static, med_trans = check_camera_static(
            video_path, start, end, yolo_model,
        )
        if is_static:
            static_windows.append((start, end))
            if verbose:
                print(f"    [cam] 窗口 #{i} 摄像机固定 ✓ (位移={med_trans:.1f}px)")
        else:
            if verbose:
                print(f"    [cam] 窗口 #{i} 摄像机移动 ✗ (位移={med_trans:.1f}px)")
    return static_windows


# ═══════════════════════════════════════════════════════════════════
# Step 6：切割 + 验证 + 降级重编码
# ═══════════════════════════════════════════════════════════════════

def _get_duration(path: Path) -> float:
    """用 ffprobe 获取文件实际时长。"""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def verify_clip_start(clip_path: Path, yolo_model) -> bool:
    """
    验证输出片段前 VERIFY_DURATION 秒的每一帧是否合格。
    用于检测 -c copy 关键帧回退引入的未扫描脏帧。
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return False

    t = 0.0
    while t <= VERIFY_DURATION:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break
        valid, _reason = evaluate_frame(frame, yolo_model)
        if not valid:
            cap.release()
            return False
        t += SAMPLE_INTERVAL

    cap.release()
    return True


def cut_clip_precise(
    video_path: Path, start: float, duration: float, output_path: Path,
) -> bool:
    """
    帧精确切割（重编码模式）。当 -c copy 引入脏帧时的降级方案。
    重编码可以在任意帧位置精确起始，不受关键帧约束。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    actual = _get_duration(output_path)
    if actual > MAX_ACTUAL_DURATION or actual < MIN_VALID_DURATION:
        output_path.unlink(missing_ok=True)
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def scan_and_segment(
    video_path: Path,
    output_dir: Path,
    video_id: str,
    yolo_model,
    scene_threshold: float = 27.0,
    verbose: bool = True,
) -> tuple[list[Path], dict]:
    """
    L3+L4 主函数：扫描视频 → 找有效窗口 → 切割 → 验证 → 输出片段。

    切割策略：
      1. 先用 -c copy 快速切割（零损耗）
      2. 验证片段开头帧（防止关键帧回退引入脏帧）
      3. 验证失败则降级为重编码精确切割

    返回:
        clip_paths: 通过验证的片段路径列表
        stats:      统计信息 {n_scenes, n_frames_scanned, n_frames_valid,
                               n_windows, n_clips, n_re_encoded}
    """
    # Step 1: 场景检测（防止切割跨硬切）
    scenes = detect_scene_boundaries(video_path, scene_threshold)
    if verbose:
        print(f"    [scan] 检测到 {len(scenes)} 个场景")

    # Step 2: 密集抽帧评估，构建时间轴（含场景屏障）
    timeline = build_frame_timeline(video_path, yolo_model, scenes)
    n_valid = sum(1 for _, v, _ in timeline if v)
    n_scanned = sum(1 for _, _, r in timeline if r != "scene_boundary")
    if verbose:
        print(f"    [scan] 评估 {n_scanned} 帧，合格 {n_valid} 帧 "
              f"({n_valid / max(n_scanned, 1) * 100:.0f}%)")

    # Step 3: 找连续有效窗口
    windows = find_valid_windows(timeline)
    if verbose:
        for i, (s, e) in enumerate(windows):
            print(f"    [scan] 窗口 #{i}: {s:.1f}s - {e:.1f}s ({e-s:.1f}s)")

    # Step 4: 摄像机运动过滤（只保留固定机位）
    n_windows_before_cam = len(windows)
    windows = filter_windows_by_camera(video_path, windows, yolo_model, verbose)
    n_cam_rejected = n_windows_before_cam - len(windows)
    if verbose and n_cam_rejected > 0:
        print(f"    [scan] 摄像机运动过滤：{n_cam_rejected}/{n_windows_before_cam} 窗口被淘汰")

    # Step 5: 切割 + 验证 + 降级
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    n_re_encoded = 0

    for i, (start, end) in enumerate(windows):
        duration = end - start
        clip_name = f"{video_id}_{i:03d}.mp4"
        out_path = output_dir / clip_name

        # 4a. 尝试 -c copy 快速切割
        if not cut_clip(video_path, start, duration, out_path):
            if verbose:
                print(f"    [scan] ✗ {clip_name} 切割失败")
            continue

        # 4b. 验证片段开头（检测关键帧回退脏帧）
        if verify_clip_start(out_path, yolo_model):
            clip_paths.append(out_path)
            if verbose:
                print(f"    [scan] ✓ {clip_name} ({duration:.1f}s)")
            continue

        # 4c. 验证失败 → 删除并用重编码精确切割
        if verbose:
            print(f"    [scan] ⚠ {clip_name} 开头有脏帧，降级重编码...")
        out_path.unlink(missing_ok=True)

        if cut_clip_precise(video_path, start, duration, out_path):
            clip_paths.append(out_path)
            n_re_encoded += 1
            if verbose:
                print(f"    [scan] ✓ {clip_name} ({duration:.1f}s, 重编码)")
        else:
            if verbose:
                print(f"    [scan] ✗ {clip_name} 重编码也失败")

    stats = {
        "n_scenes":         len(scenes),
        "n_frames_scanned": n_scanned,
        "n_frames_valid":   n_valid,
        "n_windows_raw":    n_windows_before_cam,
        "n_cam_rejected":   n_cam_rejected,
        "n_windows":        len(windows),
        "n_clips":          len(clip_paths),
        "n_re_encoded":     n_re_encoded,
    }
    return clip_paths, stats
