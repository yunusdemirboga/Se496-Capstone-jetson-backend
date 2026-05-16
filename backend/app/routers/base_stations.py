from fastapi import  APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.base_station import BaseStation
from app.schemas.base_station import BaseStationCreate, BaseStationResponse, BaseStationUpdate
from app.models.user import User
from app.dependencies import get_current_user
from typing import List
import uuid

router = APIRouter(prefix="/base_stations", tags=["base-stations"])

@router.get("/", response_model=List[BaseStationResponse])
def get_base_stations(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(BaseStation).all()

@router.post("/", response_model=BaseStationResponse)
def create_base_station(station_data: BaseStationCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    existing = db.query(BaseStation).filter(BaseStation.name == station_data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Base station with this name already exists")
    new_station = BaseStation(name=station_data.name, latitude=station_data.latitude, longitude=station_data.longitude)
    db.add(new_station)
    db.commit()
    db.refresh(new_station)
    return new_station

@router.patch("/{station_id}", response_model=BaseStationResponse)
def update_base_station(station_id: str, update_data: BaseStationUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    station = db.query(BaseStation).filter(BaseStation.id == uuid.UUID(station_id)).first()
    if not station:
        raise HTTPException(status_code=404, detail="Base station not found")
    if update_data.name is not None:
        station.name = update_data.name
    if update_data.latitude is not None:
        station.latitude = update_data.latitude
    if update_data.longitude is not None:
        station.longitude = update_data.longitude
    db.commit()
    db.refresh(station)
    return station

@router.delete("/{station_id}", status_code=204)
def delete_base_station(station_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    station = db.query(BaseStation).filter(BaseStation.id == uuid.UUID(station_id)).first()
    if not station:
        raise HTTPException(status_code=404, detail="Base Station not found")
    db.delete(station)
    db.commit()