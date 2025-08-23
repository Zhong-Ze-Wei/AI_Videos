from pydantic import BaseModel
from typing import List, Dict

class Analytics(BaseModel):
    tension: int
    complexity: int
    pacing: int
    emotion: Dict[str, int]  # <-- 将 List[int] 改为 Dict[str, int]

class Segment(BaseModel):
    id: int
    title: str
    action: str
    styleAnalysis: List[str]
    firstFramePrompt: str
    videoPrompt: str
    imageUrl: str
    analytics: Analytics

class Story(BaseModel):
    title: str
    segments: List[Segment]