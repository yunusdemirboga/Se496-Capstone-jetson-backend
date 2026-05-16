from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import uuid

class DetectionCreate(BaseModel):
    base_station_id: uuid.UUID
    drone_detected: bool
    yolo_confidence: Optional[float] = None
    acoustic_confidence: Optional[float] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    detected_at: datetime

class DetectionListResponse(BaseModel):
    id: uuid.UUID
    base_station_id: uuid.UUID
    drone_detected: bool
    image_url: Optional[str] =None
    detected_at: datetime

    class Config:
        from_attributes = True

class DetectionDetailResponse(BaseModel):
    id: uuid.UUID
    base_station_id: uuid.UUID
    drone_detected: bool
    yolo_confidence: Optional[float] = None
    acoustic_confidence: Optional[float] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    detected_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True

class DetectionListPaginated(BaseModel):
    total: int
    detections: List[DetectionListResponse]

    class Config:
        from_attributes = True