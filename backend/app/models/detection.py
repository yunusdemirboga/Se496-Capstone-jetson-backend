import uuid
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base

class Detection(Base):
    __tablename__= "detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    base_station_id = Column(UUID(as_uuid=True), ForeignKey("base_stations.id"), nullable=False)
    drone_detected = Column(Boolean, nullable=False)
    yolo_confidence = Column(Float, nullable=True)
    acoustic_confidence = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    image_url = Column(String, nullable=True)
    detected_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    base_station = relationship("BaseStation", back_populates="detections")