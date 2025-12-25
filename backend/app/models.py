from pydantic import BaseModel, Field
from typing import List, Optional

class ControlPoint(BaseModel):
    px: float
    py: float
    lat: float
    lon: float

class PlanRequest(BaseModel):
    start_address: str
    loop: bool = True
    end_address: Optional[str] = None
    max_distance_m: Optional[int] = Field(default=None, ge=200)
    allow_private: bool = True
    control_points: List[ControlPoint] = Field(min_length=2, max_length=6)

    proximity_m: float = 15.0
    max_angle_deg: float = 15.0
    completion_ratio: float = 0.90
