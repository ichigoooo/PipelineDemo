

ANALYZE_VIDEO_AGENT = """
     你是一个为人体运动数据采集做质检的专家，负责判断我发给你的视频帧是否满足以下严格要求：
     - 画面中应当只有 1 个人物为主要主体，不能有多人同时参与镜头前的动作, 背景路人较远可以忽略, 当做正常数据
     - 该人物需要全身清晰可见，从头到脚都在画面中，大部分时间都不能被裁掉或严重遮挡
     - 视频帧摄像机视角应当正常，不是剧烈的晃动、推拉、横移或频繁变焦的画面
     - 视频帧不应该有明显的剪辑、跳切、转场效果
     - 画面清晰，不能过暗、过曝或严重模糊
     - 人体的运动应当是日常动作或运动动作（例如走路、跑步、体操、舞蹈、健身等, 站立也算)
"""

ANALYZE_OUTPUT_JSON = """
    要明确区分 comment 和 description：
    - comment：必须基于“上面列出的采集要求”进行总结，说明该片段是否符合采集标准及原因。
    - description：只能基于视频内容本身进行客观描述，不得引用或暗示采集要求，不得包含是否合格的判断，并且必须且只能用一句话表达（不得换行）。
    
    这是视频中的一个视频帧，请你判断是否满足上面列出的采集要求。
    请你严格按照以下 JSON 结构给出结论（不要输出额外内容）。若 problems 为空，则不用填写该字段：
    {
    "passed": 值为 "pass" 或 "fail" ,
    "problems": [
        "no_full_body",            // 如果人体躯干没有大致完整出现在画面中 (脚或手超出画面一点点是允许的)
        "multiple_person",         // 如果画面中有多个人物参与动作或难 以分辨唯一主体
        "camera_moving",           // 如果摄像机有明显移动或剧烈晃动
        "editing_cut",             // 如果该时间范围内存在明显剪辑或跳切
        "low_quality",             // 如果画面过暗、过曝或严重模糊
        "static_pose",             // 如果人物呈现躺着或坐着不动，没有明显的运动及运动准备动作 (站立着没问题)
        "other"                    // 其他不满足采集要求的原因
    ],
    "comment": "用一两句话总结该片段是否符合采集要求及原因（必须基于采集标准）",
    "description": "仅用一句话客观描述视频内容（必须且只能一句，且字数在20字以内，不得包含合格性判断）"
    }
    你返回的文本必须严格按照 JSON 结构, 不需要用 ```json ``` 包裹内容，不要输出其他内容，不要输出任何解释说明。
    """

def build_messages(img_base64: str) -> list:
    """
    构建视频帧分析消息
    """
    if img_base64 is None:
        print("img_base64 为 空")
        return None
    
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": ANALYZE_VIDEO_AGENT},  #
                {"type": "text", "text": ""},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{ANALYZE_OUTPUT_JSON}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ],
        },
    ]
    return messages

if __name__ == "__main__":
    from motion_qc.vlm.chatbot import ChatBot
    from motion_qc.vlm.config import llm_config

    chatbot = ChatBot(llm_config["general"])

