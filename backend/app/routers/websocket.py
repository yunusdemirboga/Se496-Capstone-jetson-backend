from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.detection import Detection
from app.schemas.detection import DetectionCreate
import uuid

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@router.websocket("/ws/detections")
async def detection_websocket(websocket: WebSocket):
    await websocket.accept()
    db: Session = SessionLocal()
    try:
        while True:
            data = await websocket.receive_json()
            detection_data = DetectionCreate(**data)
            new_detection = Detection(
                base_station_id=detection_data.base_station_id,
                drone_detected=detection_data.drone_detected,
                yolo_confidence=detection_data.yolo_confidence,
                acoustic_confidence=detection_data.acoustic_confidence,
                description=detection_data.description,
                image_url=detection_data.image_url,
                detected_at=detection_data.detected_at
            )
            db.add(new_detection)
            db.commit()
            db.refresh(new_detection)
            await manager.broadcast({
                "type": "new_detection",
                "detection_id": str(new_detection.id)
            })
            await websocket.send_json({"status": "ok", "detection_id": str(new_detection.id)})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"status": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        db.close()


@router.websocket("/ws/feed")
async def feed_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)