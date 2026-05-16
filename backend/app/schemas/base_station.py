from pydantic import BaseModel
from datetime import datetime
import uuid
from typing import Optional

class BaseStationCreate(BaseModel):
    name: str
    latitude: float
    longitude: float

class BaseStationResponse(BaseModel):
    id: uuid.UUID
    name: str
    latitude: float
    longitude: float
    created_at: datetime

    class Config:
        from_attributes =True

class BaseStationUpdate(BaseModel):
    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None