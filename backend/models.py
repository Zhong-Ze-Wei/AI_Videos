from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Analytics(BaseModel):
    tension: float
    complexity: float
    pacing: float
    emotion: Dict[str, float]

class Segment(BaseModel):
    id: int
    title: str
    action: str
    styleAnalysis: List[str] = Field(default_factory=list)
    firstFramePrompt: str
    videoPrompt: str
    imageUrl: Optional[str] = None # 这个字段现在可以是空的
    analytics: Analytics

class Story(BaseModel):
    title: str
    segments: List[Segment] = Field(default_factory=list)