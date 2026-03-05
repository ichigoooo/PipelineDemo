"""
VLM 视频帧质检输出结构
"""

from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, model_validator


# 问题枚举
class Problem(str, Enum):
    no_full_body = "no_full_body"
    multiple_person = "multiple_person"
    camera_moving = "camera_moving"
    editing_cut = "editing_cut"
    occlusion = "occlusion"
    low_quality = "low_quality"
    static_pose = "static_pose"
    other = "other"


class FrameCheckResult(BaseModel):
    """视频帧质检结果"""
    passed: Literal["pass", "fail"]
    problems: Optional[List[Problem]] = None
    comment: str
    description: str

    # fail 必须有 problems
    @model_validator(mode="after")
    def validate_logic(self):
        if self.passed == "fail" and not self.problems:
            raise ValueError("fail 时必须提供 problems")
        return self

    # 从 JSON 字符串直接构造
    @classmethod
    def from_json(cls, json_str: str) -> "FrameCheckResult":
        return cls.model_validate_json(json_str)

    # 转 dict
    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    # 转 json
    def to_json(self) -> str:
        return self.model_dump_json()
    
    # 转 annotation kwargs
    def to_annotation_kwargs(self) -> dict:
        return {
            "passed": self.passed,
            "problems": [p.value for p in self.problems] if self.problems else None,
            "comment": self.comment,
            "description": self.description,
    }