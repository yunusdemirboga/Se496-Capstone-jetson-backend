from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.detection import Detection
from app.schemas.detection import DetectionListResponse, DetectionDetailResponse, DetectionListPaginated
from app.dependencies import get_current_user
from app.models.user import User
from app.services.storage import generate_signed_url
from typing import List, Optional
from datetime import datetime
import uuid

router = APIRouter(prefix="/detections", tags=["detections"])

@router.get("/", response_model=DetectionListPaginated)
def get_detections(
    base_station_id: Optional[uuid.UUID] = None,
    drone_detected: Optional[bool] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)):

    query = db.query(Detection)

    if base_station_id:
        query = query.filter(Detection.base_station_id == base_station_id)
    if drone_detected is not None:
        query = query.filter(Detection.drone_detected == drone_detected)
    if from_date:
        query = query.filter(Detection.detected_at >= from_date)
    if to_date:
        query = query.filter(Detection.detected_at <= to_date)

    total = query.count()
    detections = query.order_by(Detection.detected_at.desc()).offset(offset).limit(limit).all()

    for detection in detections:
        if detection.image_url:
            detection.image_url = generate_signed_url(detection.image_url)

    return {"total": total, "detections": detections}


@router.get("/{detection_id}", response_model=DetectionDetailResponse)
def get_detection(detection_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    detection = db.query(Detection).filter(Detection.id == uuid.UUID(detection_id)).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    if detection.image_url:
        detection.image_url = generate_signed_url(detection.image_url)
    return detection

@router.delete("/{detection_id}", status_code=204)
def delete_detection(detection_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    detection = db.query(Detection).filter(Detection.id == uuid.UUID(detection_id)).first()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    db.delete(detection)
    db.commit()