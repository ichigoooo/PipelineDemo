# 人体动作视频采集 Pipeline 技术规格书

> 文档版本：v1.0
> 最后更新：2026-03-05
> 适用范围：4000小时人体动作视频训练数据采集项目

---

## 1. 项目概述

### 1.1 目标
从原始视频库中筛选出符合机器人训练数据要求的片段，交付格式为 SQLite + OSS 文件存储。

### 1.2 数据质量标准（六条铁律）

| 序号 | 要求 | 检测层级 | 实现方式 |
|------|------|----------|----------|
| 1 | 时长 5-20s，无剪辑跳切 | L3+L4 | PySceneDetect + 连续窗口提取 |
| 2 | 分辨率 ≥720p | L1 | ffprobe 元数据检查 |
| 3 | 画面仅有单人 | L2, L3+L4, L5 | YOLO三分类 + 逐帧扫描 + VLM复核 |
| 4 | 摄像机固定，无平移/推拉/晃动 | L3+L4 | 背景光流摄像机运动检测 |
| 5 | 人体从头到脚完整可见 | L3+L4 | YOLO Pose 关键点可见性检查 |
| 6 | 横屏人高 > 画面高 1/2 | L3+L4 | 人体bbox占比计算 |

### 1.3 规模与成本

- **输入规模**：约 229,000 个视频
- **产出目标**：4,000 小时有效片段
- **VLM 调用**：约 372 万次（¥0.001/次）
- **总预算**：¥10,000（截止 2026-03-31）

---

## 2. 系统架构

### 2.1 六层过滤漏斗

```
原始视频
    │
    ├─[L0] 批次接入         ingest.py：URL列表 → 本地文件
    ├─[L1] 元数据过滤        metadata_filter.py：ffprobe筛除
    ├─[L2] 视频级YOLO预筛    prescreen.py：5帧抽样，SINGLE/DUAL/REJECT
    ├─[L3+L4] 扫描与分割     scan_and_segment.py：密集抽帧 + 窗口提取 + 摄像机检测
    ├─[L5] VLM质检          qc_runner.py：motion_qc单帧验证
    └─[L6] 交付打包         delivery.py：SQLite + OSS上传
```

### 2.2 模块依赖图

```
run_pipeline.py (主流程)
    ├── metadata_filter.py (L1)
    ├── prescreen.py (L2)
    ├── scan_and_segment.py (L3+L4)
    │       ├── clip_filter.py (工具函数)
    │       └── segmenter.py (FFmpeg切割)
    └── qc_runner.py (L5)
```

---

## 3. 各层技术规格

### 3.1 L1: 元数据过滤 (metadata_filter.py)

**功能**：快速排除不符合基本要求的视频，零解码成本。

**过滤条件**：
```python
MIN_WIDTH = 1280          # 宽度 ≥ 720p (实际要求1280)
MIN_HEIGHT = 720          # 高度 ≥ 720
MIN_DURATION = 5.0        # 时长 ≥ 5s
MAX_DURATION = 300.0      # 时长 ≤ 5分钟 (超长视频暂不处理)
REJECTED_CODECS = {'hevc', 'vp9', 'av1'}  # 暂不支持的编码
```

**实现**：调用 `ffprobe -v error -show_streams -print_format json`

---

### 3.2 L2: 视频级YOLO预筛 (prescreen.py)

**功能**：5帧均匀抽样，快速分类视频类型。

**抽样策略**：
- 帧位置：`[N/6, 2N/6, 3N/6, 4N/6, 5N/6]`（N为总帧数）
- 跳过首尾各1/6，避免片头片尾干扰

**三分类逻辑**：
```python
if 单人帧数 >= 3:   → VideoVerdict.SINGLE  → 进入L3+L4正常流程
if 双人帧数 >= 3:   → VideoVerdict.DUAL    → 标记为裁剪候选（当前未启用）
else:               → VideoVerdict.REJECT  → 丢弃
```

**模型**：YOLOv8n-pose（Nano级，速度优先）

**置信度阈值**：`YOLO_PERSON_CONF = 0.5`

---

### 3.3 L3+L4: 扫描与分割 (scan_and_segment.py)

**核心创新**：原设计（先切割再过滤）改为密集扫描后提取连续有效窗口，出材率提升约30%。

#### 3.3.1 处理流程

```
输入视频
    ↓
Step 1: PySceneDetect 场景边界检测
    ↓
Step 2: 每个场景内每0.5s密集抽帧
    ↓
Step 3: 逐帧评估（画质 + 单人 + 全身可见）
    ↓
Step 4: 构建布尔时间轴 [T,T,T,F,F,T,T,T...]
    ↓
Step 5: 场景边界插入False屏障
    ↓
Step 6: 提取连续True区间（≥5s）→ 有效窗口
    ↓
Step 7: 窗口均匀切割（>15s则拆分）
    ↓
Step 8: 摄像机运动检测（背景光流）
    ↓
Step 9: FFmpeg切割（-c copy → 验证 → 重编码降级）
    ↓
输出片段
```

#### 3.3.2 逐帧评估标准 (evaluate_frame)

**条件1：画质检查**
```python
BLUR_THRESHOLD = 50.0          # Laplacian方差阈值
BRIGHTNESS_MIN = 30            # 灰度均值下限
BRIGHTNESS_MAX = 225           # 灰度均值上限
```

**条件2：单人检测**
```python
YOLO_PERSON_CONF = 0.5         # 人体检测置信度
人数 == 1                       # 恰好1人通过，0人或≥2人拒绝
```

**条件3：全身可见性 (full_body_check)**
```python
# 关键点索引（COCO格式）
HEAD_KP_INDICES = [0, 1, 2]     # 鼻、左眼、右眼（任意1个可见）
FEET_KP_INDICES = [15, 16]      # 左踝、右踝（任意1个可见）
KP_CONF_THRESHOLD = 0.3         # 关键点置信度阈值

# 人体占比（横屏）
HEIGHT_RATIO_LANDSCAPE = 0.50   # 人高 ≥ 画面高的50%
HEIGHT_RATIO_PORTRAIT = 0.33    # 人高 ≥ 画面高的33%（竖屏）
```

#### 3.3.3 窗口提取算法 (find_valid_windows)

**连续窗口规则**：
- 起始点 = 第一个合格帧的时间戳
- 结束点 = 最后一个合格帧的时间戳 + SAMPLE_INTERVAL
- 最小窗口：`MIN_VALID_DURATION = 5.0s`
- 最大窗口：`MAX_VALID_DURATION = 15.0s`（超过则均匀切割）

**场景屏障机制**：
在相邻场景之间插入 `(scene_end, False, "scene_boundary")`，确保窗口不会跨越硬切。

#### 3.3.4 关键参数

```python
SAMPLE_INTERVAL = 0.5           # 抽帧间隔（秒）
MIN_VALID_DURATION = 5.0        # 最短有效窗口（秒）
MAX_VALID_DURATION = 15.0       # 最长有效窗口（秒）
MAX_ACTUAL_DURATION = 20.0      # 输出片段实际上限（秒）
VERIFY_DURATION = 3.0           # 切割后验证时长（秒）
```

---

### 3.4 摄像机运动检测 (Step 5)

**新增功能（2026-03-05）**：基于背景光流的固定机位验证，替代VLM单帧判断。

#### 3.4.1 算法原理

静态摄像机下，背景特征点在两帧之间应该没有全局位移；跟踪/运镜则会产生一致的背景平移。

#### 3.4.2 实现步骤

```python
def check_camera_static(video_path, window_start, window_end, yolo_model):
    translations = []

    # 每隔 CAMERA_CHECK_INTERVAL 取一个帧对
    for t in range(window_start, window_end, CAMERA_CHECK_INTERVAL):
        # 读取间隔 CAMERA_SAMPLE_GAP 的两帧
        frame1 = read_at(t)
        frame2 = read_at(t + CAMERA_SAMPLE_GAP)

        # 生成人体前景mask（bbox扩展25%）
        fg_mask = make_person_mask(frame1, yolo_model, pad=0.25)
        bg_mask = invert(fg_mask)

        # 背景区域检测Shi-Tomasi特征点
        features = goodFeaturesToTrack(gray1, mask=bg_mask, maxCorners=300)

        # Lucas-Kanade光流追踪
        tracked = calcOpticalFlowPyrLK(gray1, gray2, features)

        # RANSAC估计仿射变换（过滤局部运动干扰）
        A, inliers = estimateAffinePartial2D(good_prev, good_curr)

        # 提取平移量
        tx, ty = A[0,2], A[1,2]
        translations.append(sqrt(tx^2 + ty^2))

    # 中位数阈值判断
    median_trans = median(translations)
    return median_trans < CAMERA_TRANS_THRESH, median_trans
```

#### 3.4.3 关键参数

```python
CAMERA_SAMPLE_GAP = 0.5         # 帧对间隔（秒）
CAMERA_CHECK_INTERVAL = 1.0     # 窗口内采样频率（秒）
CAMERA_TRANS_THRESH = 2.0       # 平移阈值（像素，0.5s间隔）
CAMERA_MIN_FEATURES = 15        # 最少背景特征点数
CAMERA_BBOX_PAD = 0.25          # 人体mask扩展比例
```

#### 3.4.4 实测效果

| 场景类型 | 背景位移 | 判定结果 |
|----------|----------|----------|
| 跑步跟踪镜头 | 6-42 px | 淘汰 ✗ |
| 固定机位展示 | ~0.01 px | 保留 ✓ |
| 阈值分离度 | 300× | 鲁棒 |

---

### 3.5 切割与降级策略

**双模式切割**：

1. **快速模式** (`cut_clip`)：
   ```bash
   ffmpeg -ss {start} -i {input} -t {duration} -c copy -avoid_negative_ts 1 {output}
   ```
   - 优点：零损耗、速度快
   - 风险：关键帧对齐可能导致起点回退，引入未扫描帧

2. **精确模式** (`cut_clip_precise`)：
   ```bash
   ffmpeg -ss {start} -i {input} -t {duration} -c:v libx264 -crf 18 -preset fast -c:a aac -b:a 128k {output}
   ```
   - 触发条件：快速模式输出经 `verify_clip_start` 验证失败
   - 优点：帧精确，无关键帧约束
   - 缺点：重编码，速度较慢

**验证机制**：
- 对快速模式输出的前 `VERIFY_DURATION=3s` 逐0.5s抽帧用YOLO验证
- 发现脏帧则删除并降级为精确模式重切
- 实测重编码率：44%（教学内容，I帧间距大）

---

### 3.6 L5: VLM质检 (qc_runner.py)

**功能**：单帧最终复核，使用 motion_qc 工具。

**调用模型**：
- 默认：qwen3.5-flash via DashScope
- 本地备选：Ollama(qwen3.5:35b-a3b) / vLLM(Qwen3.5-35B-A3B-FP8)

**输出格式**：
```python
@dataclass
class FrameCheckResult:
    passed: bool          # 是否通过
    problems: List[str]   # 问题列表
    comment: str          # 评价说明
    description: str      # 图片描述
```

**成本**：~¥0.001/次

---

## 4. 输出规范

### 4.1 文件结构

```
output/
├── clips/
│   └── {video_id}/
│       ├── {video_id}_000.mp4
│       ├── {video_id}_001.mp4
│       └── ...
└── pipeline_report.json
```

### 4.2 报告格式 (pipeline_report.json)

```json
{
  "timestamp": "2026-03-05T12:34:56",
  "total_videos": 5,
  "l2_rejected": 0,
  "total_segments": 2,
  "total_clips_pass": 2,
  "videos": [
    {
      "video_id": "xxx",
      "filename": "xxx.mp4",
      "l1_pass": true,
      "l2_verdict": "single",
      "n_segments": 6,
      "n_clips_pass_l4": 2,
      "clips_pass": ["path/to/clip_000.mp4", ...]
    }
  ]
}
```

### 4.3 交付格式 (L6)

**SQLite数据库**：`{batch}.db`
```sql
CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    batch_id TEXT,
    category TEXT,
    subcategory TEXT,
    video_id TEXT,
    clip_id TEXT,
    clip_path TEXT,        -- OSS路径
    duration REAL,
    passed_l5 BOOLEAN,
    vlm_problems TEXT,
    created_at TIMESTAMP
);
```

**OSS存储**：`{batch}/{cat}/{subcat}/{vid_id}/{clip}.mp4`

---

## 5. 性能指标

### 5.1 处理速度（实测）

| 阶段 | 耗时/视频 | 说明 |
|------|-----------|------|
| L1 元数据 | <10ms | ffprobe |
| L2 YOLO预筛 | ~500ms | 5帧推理 |
| L3+L4 扫描分割 | 5-20s | 含摄像机检测 |
| L5 VLM质检 | ~2s | 网络API调用 |
| **总计** | **~30s/视频** | 单线程 |

### 5.2 出材率（测试集）

| 测试批次 | 输入时长 | 输出片段 | 出材率 |
|----------|----------|----------|--------|
| 修复前（Bug #1+#2） | 9.7 min | 17片段/3.0min | 30.9% |
| 修复后 | 9.7 min | 16片段/2.6min | 27.1% |
| +摄像机检测 | 9.7 min | 2片段/0.28min | 2.9% |

**说明**：出材率下降符合预期——测试集全为跑步视频，跟踪镜头是常态；固定机位视频（健身房/瑜伽）出材率会显著提高。

### 5.3 资源消耗

| 资源 | 用量 | 说明 |
|------|------|------|
| GPU | ~2GB VRAM | YOLOv8n-pose |
| CPU | 单核满载 | OpenCV光流计算 |
| 内存 | ~500MB | 视频解码缓冲 |

---

## 6. 关键参数汇总

### 6.1 可调节参数

| 参数 | 当前值 | 作用 | 调节方向 |
|------|--------|------|----------|
| `YOLO_PERSON_CONF` | 0.5 | 人体检测置信度 | ↑更严格，↓更宽松 |
| `BLUR_THRESHOLD` | 50.0 | 清晰度阈值 | ↑更严格 |
| `SAMPLE_INTERVAL` | 0.5s | 抽帧间隔 | ↓更密集，更慢 |
| `CAMERA_TRANS_THRESH` | 2.0px | 摄像机运动阈值 | ↑更宽松 |
| `KP_CONF_THRESHOLD` | 0.3 | 关键点置信度 | ↑更严格 |
| `HEIGHT_RATIO_LANDSCAPE` | 0.50 | 人体占比 | 需与甲方确认 |

### 6.2 建议标定实验

在 Phase 0 标定阶段，每类别取30个视频，遍历参数空间，人工标注ground truth后选择最优阈值。

---

## 7. 已知限制

1. **人体占比过大**：>80%画面时背景特征点不足，摄像机检测可能误放
2. **缓慢变焦**：当前只检测平移，未检查仿射矩阵缩放分量
3. **YOLO Nano级局限**：远处小目标、遮挡、运动模糊时检出率下降
4. **抽样盲区**：0.5s间隔可能漏过<0.5s的瞬时污染
5. **双人裁剪**：当前版本未启用，损失约10%出材率提升机会

---

## 8. 附录

### 8.1 代码文件清单

```
pipeline/
├── run_pipeline.py              # 主流程入口
├── worker/
│   ├── metadata_filter.py       # L1: 元数据过滤
│   ├── prescreen.py             # L2: YOLO预筛
│   ├── scan_and_segment.py      # L3+L4: 扫描分割（核心）
│   ├── clip_filter.py           # 工具函数库
│   ├── segmenter.py             # FFmpeg切割
│   └── qc_runner.py             # L5: VLM质检
```

### 8.2 依赖版本

```
python >= 3.10
opencv-python >= 4.8.0
ultralytics >= 8.0.0
scenedetect >= 0.6.0
numpy >= 1.24.0
```

### 8.3 运行命令

```bash
# 完整流程（含VLM）
python pipeline/run_pipeline.py \
    --input test_videos \
    --output pipeline/output \
    --yolo-model yolov8n-pose.pt

# 跳过VLM（仅L1-L4）
python pipeline/run_pipeline.py \
    --input test_videos \
    --output pipeline/output \
    --yolo-model yolov8n-pose.pt \
    --skip-vlm
```

---

*文档结束*
