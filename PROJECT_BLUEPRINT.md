# 公益机器人视频数据采集项目 — 确认基线文档

> 最后更新：2026-03-05（v4 — 明确人工验收金标准，切换为甲方供数模式）
> 本文档记录所有已确认的需求、技术决策、成本评估，作为后续开发的唯一基准。

---

## 一、项目概况

| 项目 | 内容 |
|------|------|
| 目标 | 交付 4000 小时人体动作视频训练数据 |
| 截止日期 | 2026-03-31（剩余约 28 天） |
| 预算上限 | ¥10,000 人民币 |
| 交付节奏 | 分批次，每周约 1000 小时 |
| 首批特殊规则 | 允许使用开源学术数据集作为交付内容 |

---

## 二、数据需求清单（4000 小时）

| 场景类别 | 需求小时数 | 开源覆盖度 | 主要数据来源 |
|---------|----------|----------|------------|
| 舞蹈艺术 | 1000h | 极低 | 甲方提供原始视频池 |
| 常规行走与奔跑 | 500h | 中 | 甲方提供原始视频池 |
| 体育竞技 | 500h | 中 | 甲方提供原始视频池 |
| 武术与格斗 | 500h | 低 | 甲方提供原始视频池 |
| 负重、搬运、推拉 | 400h | 低 | 甲方提供原始视频池 |
| 日常抓取 | 300h | 高（NTU） | 甲方提供原始视频池 |
| 常见工具与家务 | 300h | 高（NTU） | 甲方提供原始视频池 |
| 其他风格行走 | 200h | 低 | 甲方提供原始视频池 |
| 复杂地形/地形适应 | 200h | 极低 | 甲方提供原始视频池 |
| 爬行跳跃移动 | 100h | 低 | 甲方提供原始视频池 |
| **合计** | **4000h** | | |

---

## 三、视频质量硬性要求（六条铁律）

1. **时长**：单片段 5–20s，中间无剪辑
2. **分辨率**：≥720p（横屏 ≥1280×720，竖屏 ≥720×1280）
3. **人数**：画面中仅有且必须只有**单人**（远处背景路人可忽略）
4. **机位**：摄像机绝对静止，无平移/推拉/晃动
5. **完整度**：人体从头到脚完整可见，脚部与地面接触清晰
6. **占比**：横屏人高 > 画面高度 1/2；竖屏人高 > 画面高度 1/3

---

## 四、验收金标准：人工验收（motion_qc 仅辅助）

**核心结论：最终验收金标准是人工验收；`motion_qc_v0.1.0` 只用于辅助筛选与初检，不作为最终唯一判定。**

### 工作原理

```
视频 → FFmpeg 取第1秒关键帧（720px宽，JPEG q=8）
     → BASE64编码
     → 系统提示词 + 图像 → qwen3.5-flash API（单帧、单次调用）
     → JSON 响应 → Pydantic 验证 → FrameCheckResult
```

> **注意**：motion_qc 对每个视频仅分析**第1秒的单帧**。
> 这意味着：相机运动和剪辑跳切的检测能力有限（单帧信息不足），
> 因此其输出用于“机器预判”，最终是否合格必须以人工复核结论为准。

### VLM 后端选项（config.py 提供三种）

| 名称 | 模型 | Base URL | 适用场景 |
|------|------|----------|---------|
| `general` | qwen3.5-flash | DashScope API | **默认**，低成本，高速 |
| `ollama` | qwen3.5:35b-a3b | 本地 Ollama | 本地 GPU 部署 |
| `ucloud` | Qwen3.5-35B-A3B-FP8 | vLLM | 云端 GPU 部署 |

### 判定问题类型

| 问题代码 | 含义 | 我们的前置过滤能否提前拦截？ |
|---------|------|------------------------|
| `multiple_person` | 多人参与动作 | ✅ YOLO 可靠拦截 |
| `no_full_body` | 人体不完整 | ⚠️ YOLO 有限（不做） |
| `camera_moving` | 摄像机明显移动 | ❌ 单帧无法判断 |
| `editing_cut` | 剪辑跳切 | ✅ PySceneDetect 拦截 |
| `low_quality` | 过暗/过曝/模糊 | ⚠️ OpenCV 可粗筛 |
| `static_pose` | 躺/坐着不动 | ❌ 不做 |
| `other` | 其他 | ❌ 仅 VLM 判断 |

### 输出结构（FrameCheckResult）

```python
passed: "pass" | "fail"
problems: Optional[List[Problem]]   # fail 时必须非空
comment: str                        # 基于采集标准的判断说明
description: str                    # ≤20字客观内容描述（单句）
```

---

## 五、交付格式规范

### 批次交付物结构

```
{批次名}.db                            ← SQLite3 数据库
{批次名}/                               ← 仅含 pass 的视频文件
    └── {类别}/
        └── {子类别}/
            └── {源视频ID}/
                └── {片段文件名}.mp4
```

> 路径示例（来自甲方 PDF）：
> `b2_20260303/体育1/Aerobic_Gymnastics/102078_cNhgeTOt-9I_21/102078_cNhgeTOt-9I01.mp4`

### SQLite `results` 表

```sql
CREATE TABLE results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image       TEXT NOT NULL,   -- 批次内相对路径
    passed      TEXT NOT NULL,   -- 固定 "pass"
    problems    TEXT,            -- NULL（pass 时不填）
    comment     TEXT NOT NULL,   -- VLM 评价
    description TEXT NOT NULL    -- ≤20字内容描述
);
CREATE UNIQUE INDEX idx_image ON results(image);
```

> **规则**：DB 文件中只写入 `pass` 记录，不出现 `fail` 行。
> 只上传 `pass` 的视频文件。建议上传前人工复核一遍 pass 文件列表。

---

## 六、技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| 开发环境 | WSL2 (Ubuntu) | 本地开发、调试 |
| 调度中心 | 轻量云服务器 + FastAPI + PostgreSQL | 任务队列、进度追踪 |
| Worker 节点 | AutoDL RTX 3090/4090 + Docker | GPU 推理 + 视频处理 |
| 数据接入 | 本地文件扫描 + 批次清单校验 | 接收甲方提供视频，不做网络爬取 |
| 场景切割 | PySceneDetect | 检测剪辑点（非运镜） |
| 人体预过滤 | YOLO-Pose (yolov8n-pose) | 仅做单人检测 |
| 画质预筛 | OpenCV (Laplacian + 直方图) | 模糊/曝光快速检测 |
| VLM 质检 | motion_qc_v0.1.0 | 单帧验收（qwen3.5-flash） |
| 对象存储 | 阿里云 OSS | 数据交付 |

---

## 七、七层过滤漏斗（核心架构）

设计原则：**越早丢弃成本越低。每一层只拦截自己最擅长检测的问题。**

### 漏斗总览

```
甲方原始视频池
    ↓
[Layer 0] 批次接入 — 清单校验、完整性检查
    ↓                                         成本: ¥0
[Layer 1] 元数据过滤 — 分辨率/时长/编码过滤
    ↓ 过滤 ~20-30%                            成本: ¥0
[Layer 2] 视频级预筛 — 整视频 YOLO 抽样（单人放行/双人标记/其他丢弃）
    ↓ 丢弃 ~30%，标记双人候选 ~10%            成本: 极低 (GPU 微秒级)
[Layer 3] 切割与机械过滤 — PySceneDetect + 时长
    ↓ 产出片段池                               成本: 低 (CPU)
[Layer 4] 片段级预筛 — YOLO单人 + 双人裁剪放大 + 模糊/曝光
    ↓ 过滤 ~35%（双人片段裁剪为2个单人片段）    成本: 低 (GPU 微秒级)
[Layer 5] VLM 质检 — motion_qc 辅助判定
    ↓ 过滤 ~35-65%                            成本: ¥0.001/次
[Layer 6] 人工复核 + 交付管线 — SQLite + OSS 上传
    ↓
  有效片段入库
```

---

### 7.0 Layer 0：批次接入（零成本，最高杠杆）

**目的**：将甲方提供的视频批次稳定接入本地处理池，先做清点再开跑漏斗。

| 检查项 | 规则 | 失败处理 |
|------|------|--------|
| 批次清单 | 必须包含唯一 `source_video_id` 与文件相对路径 | 标记缺失并回传甲方补齐 |
| 文件可读性 | `ffprobe` 可解析，时长 > 0 | 进入坏文件清单，不进入漏斗 |
| 去重 | 同一 `source_video_id` 仅保留一份有效文件 | 重复项仅保留首个并记录 |
| 目录映射 | 映射到目标类别/子类别（可用甲方标签） | 未命中类别进入待人工标注池 |

**关键原则**：
- 本项目不承担互联网检索、爬取和下载职责
- 甲方供数质量波动要前置暴露，避免脏数据进入后续高成本阶段
- 接入层只做事实校验，不做语义质量判断

---

### 7.1 Layer 1：元数据过滤（近零成本）

**目的**：在进入计算密集环节前，通过基础元数据过滤明显不合格视频，节省算力和存储。

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,codec_name \
  -show_entries format=duration \
  -of json "{input_video}"
```

**过滤规则建议**：
- 分辨率：短边必须 ≥ 720
- 时长：`5s <= duration <= 3600s`
- 编码：优先 H.264/H.265；非常见编码先转码再入池
- 损坏文件：`ffprobe` 异常直接淘汰并记录

**过滤效果**：预计排除 ~20-30% 的甲方原始数据。

---

### 7.2 Layer 2：视频级预筛（极低成本，高杠杆）

**目的**：在进行 PySceneDetect 切割之前，快速判断整个视频是否值得处理。
防止对无效视频做昂贵的切割和逐片段分析。

**方法**：
```python
from enum import Enum

class VideoVerdict(Enum):
    SINGLE = "single"       # 单人视频，正常流程
    DUAL = "dual"           # 双人视频，进入裁剪流程
    REJECT = "reject"       # 丢弃（3人+、无人、纯风景等）

def video_prescreen(video_path, model) -> VideoVerdict:
    """从视频均匀抽取 5 帧，YOLO 检测人数，三分类。"""
    frames = extract_uniform_frames(video_path, n=5)
    counts = {1: 0, 2: 0}
    for frame in frames:
        results = model(frame)
        persons = [b for b in results.boxes if b.cls == 0 and b.conf > 0.5]
        n = len(persons)
        if n in counts:
            counts[n] += 1

    if counts[1] >= 3:
        return VideoVerdict.SINGLE       # ≥3/5 帧单人 → 正常流程
    if counts[2] >= 3:
        return VideoVerdict.DUAL          # ≥3/5 帧双人 → 标记裁剪候选
    return VideoVerdict.REJECT            # 其他 → 丢弃
```

**三分类流向**：

| 判定 | 占比（估） | 后续处理 |
|------|----------|---------|
| `SINGLE` | ~50% | 正常进入 Layer 3 切割 |
| `DUAL` | ~10-15% | 进入 Layer 3，但片段在 Layer 4 走裁剪分支 |
| `REJECT` | ~35-40% | 丢弃（3人+、无人、纯风景等） |

**性能**：5 次 YOLO 推理/视频。GPU (RTX 4090): ~0.05s/视频。CPU: ~2.5s/视频。

**价值量化**：每淘汰 1 个 REJECT 视频 = 节省 ~23 次 VLM 调用 = 节省 ~¥0.023。
每挽救 1 个 DUAL 视频 = 额外产出 ~16 个候选片段（2×8）。

---

### 7.3 Layer 3：切割与机械过滤（CPU 成本）

**目的**：将通过 Layer 2 的视频切割为连续片段，排除不满足时长要求的部分。

**流程**：
1. **PySceneDetect 场景检测**：识别视频中的所有内容突变点（剪辑切换、转场）
2. **连续片段提取**：以场景边界为切点，提取每个连续无剪辑的片段
3. **长片段均匀分割**：对 >20s 的连续片段，按 10-20s 均匀切割
   - 这不违反"无裁剪"要求：原始素材是连续的，我们只是取其中的一个时间窗口
4. **时长过滤**：丢弃 <5s 的片段

**FFmpeg 切割命令**（精确关键帧对齐）：
```bash
ffmpeg -ss {start} -i {video} -t {duration} -c copy -avoid_negative_ts 1 {output}
```

**预计产出**：每个通过 Layer 2 的视频 → 平均 ~35 个候选片段。

---

### 7.4 Layer 4：片段级预筛 + 双人裁剪（低 GPU 成本）

**目的**：用极低成本的计算机视觉方法，排除明显废片；对双人片段执行裁剪+放大变废为宝。

#### 4A. 标准路径（单人视频的片段）

```python
def clip_prescreen(clip_path, yolo_model) -> list[str]:
    """返回通过预筛的片段路径列表（0个=淘汰，1个=原片，2个=裁剪产出）"""
    frame = extract_middle_frame(clip_path)
    results = yolo_model(frame)
    persons = [b for b in results.boxes if b.cls == 0 and b.conf > 0.5]

    # 画质基础检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:     # 模糊
        return []
    if not (30 < gray.mean() < 225):                    # 过暗/过曝
        return []

    # 单人 → 直接放行
    if len(persons) == 1:
        return [clip_path]

    # 恰好双人 + 分辨率足够 → 裁剪分支
    if len(persons) == 2:
        return try_crop_dual_person(clip_path, persons, frame.shape)

    # 其他情况 → 淘汰
    return []
```

#### 4B. 双人裁剪+放大分支

**触发条件**：YOLO 检测到恰好 2 人 + 源片段分辨率 ≥1080p。

**原理**：从 1080p 视频裁剪出每人区域后，通过 FFmpeg lanczos 1.33× 放大恢复到 ≥720p。

| 原始分辨率 | 裁剪后 | 1.33× 放大后 | 满足 ≥720p？ |
|-----------|--------|-------------|------------|
| 1920×1080 | ~960×1080 | 1280×1440 | ✅ 横屏 |
| 1080×1920 (竖屏) | ~540×1920 | 720×2560 | ✅ 竖屏 |
| 1280×720 | ~640×720 | 853×960 | ❌ 需2×放大，画质损失大 |
| ≥2560×1440 | ≥1280×1440 | 无需放大 | ✅ 原生合规 |

```python
# 最大允许放大倍率（超过此值画质损失过大）
MAX_UPSCALE = 1.5

def try_crop_dual_person(clip_path, persons, frame_shape) -> list[str]:
    """尝试将双人片段裁剪为两个单人片段。"""
    h, w = frame_shape[:2]
    results = []

    for i, person in enumerate(persons):
        # YOLO bbox → 裁剪区域（加 20% padding）
        x1, y1, x2, y2 = person.xyxy[0].tolist()
        pad_w = (x2 - x1) * 0.2
        pad_h = (y2 - y1) * 0.1
        cx1 = max(0, int(x1 - pad_w))
        cy1 = max(0, int(y1 - pad_h))
        cx2 = min(w, int(x2 + pad_w))
        cy2 = min(h, int(y2 + pad_h))
        crop_w, crop_h = cx2 - cx1, cy2 - cy1

        # 计算放大倍率（需要达到 ≥1280×720 或 ≥720×1280）
        if crop_w >= crop_h:  # 横屏裁剪
            scale = max(1280 / crop_w, 720 / crop_h, 1.0)
        else:                 # 竖屏裁剪
            scale = max(720 / crop_w, 1280 / crop_h, 1.0)

        if scale > MAX_UPSCALE:
            continue  # 放大倍率过大，放弃此人

        # FFmpeg 单次 crop + scale（lanczos 插值）
        out_w, out_h = int(crop_w * scale), int(crop_h * scale)
        # 确保偶数（编码器要求）
        out_w, out_h = out_w // 2 * 2, out_h // 2 * 2
        output = generate_crop_path(clip_path, person_index=i)
        cmd = [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-vf", f"crop={crop_w}:{crop_h}:{cx1}:{cy1},"
                   f"scale={out_w}:{out_h}:flags=lanczos",
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "copy",
            str(output)
        ]
        if execute_ffmpeg(cmd):
            results.append(output)

    return results  # 0、1 或 2 个片段
```

**关键设计要点**：

1. **固定裁剪框**：从参考帧（中间帧）的 YOLO bbox 计算裁剪区域，整个片段使用固定裁剪区域。不逐帧跟踪（避免抖动被判 `camera_moving`）
2. **只用 FFmpeg lanczos**：不用 AI 超分辨率。1.33× 放大肉眼几乎无损，且 motion_qc 质检缩略图为 720px，放大前后在缩略图上无差异
3. **最大放大倍率 1.5×**：超过此值画质损失明显，放弃该裁剪
4. **CRF 18 编码**：接近无损的 H.264 编码，避免二次压缩引入的质量损失
5. **每个裁剪片段独立过 VLM**：裁剪后仍需通过 motion_qc 验收

**性能**：裁剪+放大一个 10s@1080p 片段 ≈ 1-2 秒（FFmpeg CPU 编码）。

**过滤效果**（含裁剪）：Layer 4 整体淘汰 ~30%（原 35%，因双人挽回一部分）。

---

#### Layer 4 产出汇总

| 输入类型 | 处理方式 | 产出 |
|---------|---------|------|
| 单人片段（通过画质检测） | 直接放行 | 1 个片段 |
| 双人片段（≥1080p） | 裁剪+放大 → 2 个单人片段 | 0-2 个片段 |
| 双人片段（<1080p） | 淘汰 | 0 |
| 3人+/无人/画质不合格 | 淘汰 | 0 |

---

### 7.5 Layer 5：VLM 质检 — motion_qc（主要成本，辅助判定）

**目的**：机器预判。所有通过 Layer 4 的片段调用 motion_qc 库进行 VLM 分析，为人工终验提供结构化依据。

**调用方式**（直接 import motion_qc 为 Python 库）：

```python
from motion_qc.media.thumb import gen_video_thumb
from motion_qc.vlm.chatbot import ChatBot
from motion_qc.vlm.prompt import build_messages
from motion_qc.vlm.config import llm_config
from motion_qc.types import FrameCheckResult

chatbot = ChatBot(llm_config["general"], silent=True)

def vlm_check(clip_path, thumb_dir) -> FrameCheckResult | None:
    thumb_path = gen_video_thumb(clip_path, thumb_dir / "thumb.jpg", width=720)
    if thumb_path is None:
        return None
    img_b64 = chatbot.encode_image(thumb_path.as_posix())
    msgs = build_messages(img_b64)
    resp = chatbot.chat_with_messages(msgs)
    json_str = chatbot.extract_json(resp)
    if json_str is None:
        return None
    return FrameCheckResult.from_json(json_str)
```

**关键参数**：
- 缩略图：第 1 秒关键帧，720px 宽，JPEG q=8
- 模型：qwen3.5-flash（DashScope API）
- 单次调用：1 张图片 + prompt ≈ 1000-1200 input tokens + ~180 output tokens

**VLM 单次成本估算**：

| 组成 | Tokens | 单价 | 成本 |
|------|--------|------|------|
| 文本输入 | ~350 | ¥0.4/M | ¥0.00014 |
| 图像输入（720px 缩略图） | ~400-800 | ¥0.4/M | ¥0.00016-0.00032 |
| 文本输出 | ~180 | ¥3.2/M | ¥0.000576 |
| **合计** | | | **≈ ¥0.0008-0.0012** |

> **保守估算取 ¥0.001/次作为基准**。实际成本需在标定批次后根据 DashScope 账单确认。

---

### 7.6 Layer 6：人工复核 + 交付管线

**目的**：以人工验收作为最终门禁，将“机器 pass”转化为“人工确认 pass”后再交付。

**流程**：
1. **候选集生成**：汇总 VLM `pass` 与边界样本（例如低置信度/描述不一致样本）
2. **人工终验**：按六条铁律逐条确认，人工结论覆盖机器结论
3. **文件组织**：仅对人工确认 `pass` 的片段按 `{batch}/{category}/{subcategory}/{video_id}/{clip}.mp4` 落盘
4. **SQLite 写入**：仅写入人工确认 `pass` 记录，保留 `comment` / `description`
5. **OSS 上传**：`{batch}.db` + 所有人工确认 `pass` 视频文件
6. **Master 汇报与清理**：更新进度并清理临时文件

---

## 八、关键技术决策记录

### 8.1 motion_qc 使用方式：原样调用，不修改

- motion_qc 采用单帧分析（第 1 秒），用于辅助筛选与预判
- 我们不引入 3 帧宫格或其他增强方案
- 最终验收金标准为人工复核，motion_qc 结果不能直接替代人工结论
- 原因：避免工具偏差放大；同时保留低成本自动化收益

### 8.2 YOLO-Pose 职责严格限定

```python
# Layer 2（视频级）：≥3/5 帧为单人 → 放行
# Layer 4（片段级）：中间帧为单人 → 放行
# 不判断：人体占比、触地情况、全身可见性
```

- 原因：YOLO 对占比/触地的假阴性率高，VLM 对此更可靠
- YOLO 的核心价值：极低成本排除 `multiple_person` 和无人视频

### 8.3 PySceneDetect 的能力边界

- **能做**：检测画面内容突变（硬切、转场、淡入淡出）
- **不能做**：检测摄像机平移/推拉（运镜≠场景切换）
- 运镜检测由 VLM 在第 1 秒帧的视觉线索中做辅助判断（能力有限，最终以人工复核为准）

### 8.4 类别分类策略

- **接入时确定类别**：优先使用甲方提供标签/目录映射到 10 个类别之一
- 不额外做 VLM 分类（motion_qc 的 `description` 字段可辅助确认）
- 对标签缺失样本使用规则映射 + 人工补标

### 8.6 双人裁剪+放大策略

- **触发条件**：YOLO 检测到恰好 2 人 + 源分辨率 ≥1080p
- **放大方式**：FFmpeg lanczos 插值（不用 AI 超分辨率）
- **最大放大倍率**：1.5×（超过则放弃）
- **裁剪区域**：基于参考帧 YOLO bbox + 20% padding，固定应用于全片段
- **编码**：libx264 CRF 18，接近无损
- **预期增量**：~10% 额外有效产出（~340h for Phase 2）
- **不使用 AI 超分的原因**：
  - 1.33× 放大幅度下 lanczos 已足够
  - AI 超分（Real-ESRGAN）每片段需 2-10 分钟，成本高 500×
  - motion_qc 质检缩略图为 720px，放大前后的缩略图几乎一致

### 8.7 去重策略

- 在 Master PostgreSQL 中记录已处理的 `source_video_id`（甲方唯一ID）
- 同一视频不会被多个 Worker 重复处理
- SQLite `results.image` 字段建 UNIQUE INDEX，防止片段级重复

---

## 九、漏斗成本模型（甲方供数模式）

### Phase 1（甲方存量批次）

| 项目 | 数值 |
|------|------|
| 预期可用小时 | 600-800h |
| 需处理片段数（高通过率） | ~500,000 |
| Layer 4 过滤后送 VLM | ~400,000 |
| VLM 成本 | 400,000 × ¥0.001 = **¥400** |

### Phase 2（甲方增量批次）出材率模型

**基于单视频（10 分钟，甲方供数）的流转模型**：

```
接入 1 个视频（10 min）
  │
  ├── Layer 2 分类：
  │   ├── SINGLE（50%）→ 正常流程
  │   │     ├── PySceneDetect → ~10 片段 → 时长过滤 → ~35 片段
  │   │     ├── Layer 4 标准路径（65% 放行）→ ~23 片段送 VLM
  │   │     └── Layer 5 VLM（35% pass）→ ~8 个候选片段
  │   │
  │   ├── DUAL（12%）→ 裁剪流程（≥1080p 时触发，~85% 的双人视频满足）
  │   │     ├── PySceneDetect → ~10 片段 → 时长过滤 → ~35 片段
  │   │     ├── Layer 4 裁剪分支 → 每片段裁出 ~1.6 个单人片段 → ~56 片段
  │   │     ├── Layer 4 画质过滤（65% 放行）→ ~36 片段送 VLM
  │   │     └── Layer 5 VLM（30% pass）→ ~11 个候选片段
  │   │
  │   └── REJECT（38%）→ 丢弃
  │
  └── 单视频平均候选产出：
      0.50 × 80s + 0.12 × 110s = 53.2s
      ≈ 0.89 有效 min / 接入视频
```

> 注：上式为“进入人工终验前”的候选产出。最终交付量以 Layer 6 人工验收通过率为准。

**Phase 2 总量推导**（需要 ~3,400 有效小时）：

| 指标 | 无裁剪 | 含裁剪（v4） |
|------|-------|------------|
| 需接入视频数 | 305,000 | **229,000** |
| 通过 Layer 2（非REJECT） | 189,000 | 142,000 |
| VLM 调用次数 | 4,347,000 | **3,720,000** |
| VLM 成本 | ¥4,347 | **¥3,720** |

### 总预算分配（v4，含人工终验）

| 项目 | 金额 | 说明 |
|------|------|------|
| VLM API（Phase 1 + 2） | ¥4,100 | Phase1 ¥400 + Phase2 ¥3,720 |
| GPU 算力（AutoDL） | ¥1,800 | 3 台 RTX 4090 × 10 天 × ¥60/天 |
| 调度服务器 | ¥300 | 1 核 2G × 30 天 |
| OSS 存储 + 流量 | ¥1,200 | ~5TB 存储 + 出流量 |
| 人工终验成本 | ¥1,600 | 边界样本 + 机器 pass 全量复核 |
| **合计** | **¥9,000** | |
| **安全余量** | **¥1,000** | |

---

## 十、出材率标定方案（开工前必做）

**问题**：上述出材率模型基于估算，实际可能有 ±50% 偏差。

**标定方法**：在大规模启动前，对每个类别运行小样本测试。

### 标定流程

```
对每个目标类别 (10个)：
  1. 从甲方已提供视频中随机选取 30 个视频
  2. 跑完整 Pipeline (Layer 1-5)
  3. 记录每层的淘汰率
  4. 计算该类别的端到端候选出材率
  5. 叠加人工终验通过率，推导最终交付出材率
  6. 反推完成该类别小时目标所需接入量与 VLM 成本
```

### 标定成本

- 10 类别 × 30 视频 = 300 个视频
- VLM 调用：约 300 × 23 = 6,900 次
- 成本：~¥7（可忽略）
- 时间：1 台 Worker 约 2-3 小时 + 人工终验抽样时间

### 标定产出

一张表：

| 类别 | L2淘汰率 | L4淘汰率 | VLM通过率 | 人工终验通过率 | 端到端出材率 | 完成目标需接入量 | 预估VLM成本 |
|------|---------|---------|----------|--------------|-----------|---------------|----------|
| 舞蹈 | ?% | ?% | ?% | ?% | ?% | ? 视频 | ¥? |
| ... | | | | | | | |

**此表确认后，整体预算预测误差从 ±200% 压缩到 ±20%。**

---

## 十一、执行阶段计划

### Phase 0：环境搭建 + 标定（第 1-2 天）

- [ ] 搭建本地开发环境（WSL2, Python, 依赖安装）
- [ ] 编写 Pipeline 核心代码（Layer 0-6）
- [ ] 跑通“甲方单批次”端到端测试
- [ ] 运行 10 类别标定测试（每类 30 视频）
- [ ] 根据标定结果修正预算和接入量估算

### Phase 1：甲方存量批次处理（第 2-7 天，目标 600-800h）

| 批次类型 | 预期小时 | 处理重点 |
|--------|---------|---------|
| 历史存量高质量批次 | ~400h | 快速出片，优先覆盖短缺类别 |
| 历史存量混合质量批次 | ~200h | 依赖 Layer 2-4 强过滤 |
| 标签不完整批次 | ~200h | 规则映射 + 人工补标后进入流水线 |

### Phase 2：甲方增量批次持续处理（第 7-28 天，目标 3,200-3,400h）

**分阶段扩容**：

| 时段 | Worker 数 | 目标 | 重点类别 |
|------|----------|------|---------|
| 第 7-10 天 | 1-2 台 | 调试+验证 | 舞蹈（缺口最大 1000h） |
| 第 10-18 天 | 3-5 台 | 稳定产出 | 舞蹈 + 武术 + 体育 |
| 第 18-28 天 | 3-5 台 | 冲刺补缺 | 按类别缺口动态调配 |

**每日吞吐量估算（每 Worker）**：

```
接入解析：~5,000 视频/天（受磁盘 I/O 与转码影响）
VLM 调用：~15,000 次/天（受 API 速率限制）
人工终验：~3,000-5,000 候选片段/天（受人力配置影响）
有效产出：~90-130 有效小时/天
```

---

## 十二、风险登记册

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 甲方供数质量波动大 | 中 | 出材率下降、工期拉长 | Layer 0/1 前置质量门禁 + 及时回传坏片清单 |
| qwen3.5-flash API 限流/不可用 | 低 | 产出停滞 | 切换至本地部署（Ollama/vLLM） |
| 人工终验吞吐不足 | 中 | 候选积压、延期交付 | 建立分级复核（高置信快速通道 + 边界样本优先） |
| 甲方增量批次补料延迟 | 中 | Worker 空转、进度波动 | 按批次节奏滚动排产，提前锁定下一批 |
| 甲方追加质量要求 | 低 | 返工 | 标定阶段主动确认边界案例并固化 checklist |
| 实际 VLM 单价高于估算 | 中 | 预算超支 | 标定后精算；必要时切换本地部署 |

---

## 十三、待确认事项（需与甲方沟通）

- [ ] 镜子导致视觉上出现"双人"的片段是否判定为废片
- [ ] 人物背对镜头但满足解剖结构完整的片段是否有效
- [ ] 非遮挡性平台水印是否可接受
- [ ] 人工终验抽检比例与责任边界（全检 / 抽检）如何定义
- [ ] 批次命名规则详细说明（PDF 提及"S3 对象存储数据上传手册"，尚未获得）
- [ ] 类别编号规则（如"体育1"中的数字含义）

---

## 十四、项目目录结构（规划）

```
/home/septem/Desktop/Mining/
├── PROJECT_BLUEPRINT.md          ← 本文档（基线参考）
├── requirements.md               ← 原始需求描述
├── 需求文档/
│   ├── 数据需求0303（4000）.xlsx  ← 类别需求表
│   └── 质检和交付要求.pdf         ← 交付规范
├── motion_qc_v0.1.0/             ← 辅助质检工具（只读引用，不修改）
└── pipeline/                     ← 主管线代码
    ├── config/                   ← 配置文件（批次映射、类别映射、API Key）
    ├── master/                   ← 调度中心（FastAPI + PostgreSQL）
    ├── worker/                   ← Worker 节点核心代码
    │   ├── ingest.py             ← Layer 0: 批次接入与清单校验
    │   ├── metadata_filter.py    ← Layer 1: 元数据过滤
    │   ├── prescreen.py          ← Layer 2: 视频级 YOLO 预筛
    │   ├── segmenter.py          ← Layer 3: PySceneDetect + 切割
    │   ├── clip_filter.py        ← Layer 4: 片段级预筛
    │   ├── qc_runner.py          ← Layer 5: motion_qc 调用封装
    │   └── delivery.py           ← Layer 6: 人工复核 + SQLite + OSS 上传
    ├── calibration/              ← 标定脚本
    └── monitoring/               ← 进度监控（Streamlit）
```
