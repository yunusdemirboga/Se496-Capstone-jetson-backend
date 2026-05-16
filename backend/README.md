# UAV Detection System — Backend

A FastAPI backend for a UAV (drone) detection system dashboard. The system receives detection reports from a Jetson device via WebSocket, stores them in a PostgreSQL database hosted on Supabase, and serves the data to an authorized dashboard frontend.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI 0.135 |
| Database | PostgreSQL (Supabase) |
| ORM | SQLAlchemy 2.0 |
| Migrations | Alembic |
| Authentication | JWT (python-jose) |
| Password Hashing | Argon2id (argon2-cffi) |
| Image Storage | Supabase Storage — private bucket, signed URLs |
| Realtime Ingestion | WebSocket (Jetson → backend + backend → dashboard broadcast) |

---

## Project Structure

```
Capstone-Backend/
├── app/
│   ├── main.py               # App entry point, CORS middleware, router registration
│   ├── config.py             # Environment variable settings (pydantic-settings)
│   ├── database.py           # SQLAlchemy engine, session, Base, get_db dependency
│   ├── dependencies.py       # JWT token validation dependency (get_current_user)
│   ├── models/
│   │   ├── __init__.py       # Imports all models
│   │   ├── user.py           # Users table
│   │   ├── base_station.py   # Base stations table
│   │   └── detection.py      # Detections table
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py           # User request/response schemas + Token schema
│   │   ├── base_station.py   # Base station request/response schemas
│   │   └── detection.py      # Detection request/response schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py           # /register, /login, /token, /me endpoints
│   │   ├── base_stations.py  # Base station list, create, update, delete endpoints
│   │   ├── detections.py     # Detection list, detail, delete endpoints (with filters and signed URLs)
│   │   └── websocket.py      # WebSocket endpoints: Jetson ingestion + real-time frontend broadcast
│   └── services/
│       ├── __init__.py
│       ├── auth.py           # Argon2id password hashing and JWT token logic
│       └── storage.py        # Supabase Storage signed URL generation
├── alembic/                  # Database migration files
├── .env                      # Local environment variables (never commit)
├── .env.example              # Environment variable template (safe to commit)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Database Schema

### `users`
| Column | Type | Description |
|---|---|---|
| id | UUID | Primary key, auto-generated |
| username | VARCHAR | Unique, required |
| hashed_password | VARCHAR | Argon2id hash, required |
| created_at | TIMESTAMP | Auto-set on creation (UTC) |

### `base_stations`
| Column | Type | Description |
|---|---|---|
| id | UUID | Primary key, auto-generated |
| name | VARCHAR | Unique station name, required |
| latitude | FLOAT | Physical location, required |
| longitude | FLOAT | Physical location, required |
| created_at | TIMESTAMP | Auto-set on creation (UTC) |

### `detections`
| Column | Type | Description |
|---|---|---|
| id | UUID | Primary key, auto-generated |
| base_station_id | UUID | Foreign key → base_stations.id |
| drone_detected | BOOLEAN | Whether a drone was detected |
| yolo_confidence | FLOAT | YOLO model confidence (0.0–1.0), nullable |
| acoustic_confidence | FLOAT | Acoustic model confidence (0.0–1.0), nullable |
| description | TEXT | LLM-generated report description, nullable |
| image_url | VARCHAR | File path in Supabase Storage, nullable |
| detected_at | TIMESTAMP | Timestamp from Jetson, required |
| created_at | TIMESTAMP | Auto-set on insertion (UTC) |

> **Note:** `image_url` stores the file path within the Supabase Storage bucket (not a full URL). When the frontend requests detections, the backend generates a time-limited signed URL on the fly using the Supabase Storage API.

---

## API Endpoints

### Auth
| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| POST | `/auth/register` | No | Create a new user |
| POST | `/auth/login` | No | Login and receive JWT token (JSON body) |
| POST | `/auth/token` | No | Login and receive JWT token (form data — for Swagger UI) |
| GET | `/auth/me` | Yes | Get current logged-in user |

### Base Stations
| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| GET | `/base_stations/` | Yes | List all base stations |
| POST | `/base_stations/` | Yes | Register a new base station |
| PATCH | `/base_stations/{id}` | Yes | Update a base station's name, latitude, or longitude |
| DELETE | `/base_stations/{id}` | Yes | Delete a base station |

#### `PATCH /base_stations/{id}` — Request Body

All fields are optional. Only the fields you include will be updated.

```json
{
  "name": "New Station Name",
  "latitude": 24.7136,
  "longitude": 46.6753
}
```

### Detections
| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| GET | `/detections/` | Yes | List detections (newest first), with optional filters, pagination, and total count |
| GET | `/detections/{id}` | Yes | Get full report for a single detection, with signed image URL |
| DELETE | `/detections/{id}` | Yes | Delete a detection (e.g. to remove a false positive) |

#### `GET /detections/` — Query Parameters

All parameters are optional. If none are provided, the 50 newest detections are returned.

| Parameter | Type | Description |
|---|---|---|
| `base_station_id` | UUID | Filter by a specific base station |
| `drone_detected` | boolean | `true` = drone detected only, `false` = no drone only |
| `from_date` | datetime | Only return detections at or after this date/time |
| `to_date` | datetime | Only return detections at or before this date/time |
| `limit` | integer | Number of results to return (default: 50) |
| `offset` | integer | Number of results to skip, used for pagination (default: 0) |

#### `GET /detections/` — Response Format

The response is a paginated wrapper object, not a plain list:

```json
{
  "total": 143,
  "detections": [
    {
      "id": "uuid",
      "base_station_id": "uuid",
      "drone_detected": true,
      "image_url": "https://signed-url...",
      "detected_at": "2026-03-28T10:38:50Z"
    }
  ]
}
```

- `total` — the total number of detections matching the filters (ignoring `limit`/`offset`), use this to calculate total pages: `Math.ceil(total / limit)`
- `detections` — the current page of results

**Example URLs:**
```
/detections/                                              → 50 newest detections
/detections/?limit=20&offset=20                           → results 21–40 (page 2)
/detections/?drone_detected=true                          → positive detections only
/detections/?base_station_id=543a8b49-fb4b-4e43-85f5-f60be027caf5
/detections/?from_date=2026-03-01T00:00:00&to_date=2026-03-28T23:59:59
/detections/?drone_detected=true&limit=10&offset=0        → combine filters
```

### WebSocket
| Type | Endpoint | Auth Required | Description |
|---|---|---|---|
| WebSocket | `/ws/detections` | No | Jetson connects here to send detection reports |
| WebSocket | `/ws/feed` | No | Dashboard frontend connects here to receive real-time new detection broadcasts |

---

## WebSocket — Real-Time Frontend Feed

When the Jetson sends a new detection via `/ws/detections`, the backend saves it to the database and **immediately broadcasts a notification** to all connected dashboard clients via a `ConnectionManager`.

Dashboard clients connect to `ws://<your-server>/ws/feed` to receive these broadcasts.

### Broadcast message format (sent to all frontend clients):
```json
{
  "type": "new_detection",
  "detection_id": "uuid-of-the-new-detection"
}
```

The frontend uses this signal to invalidate its local cache and re-fetch the detections list, making new detections appear on the dashboard instantly without polling.

### Flow:
```
Jetson → POST detection to /ws/detections
              ↓
         Backend saves to DB
              ↓
         Broadcasts { type: "new_detection", detection_id: "..." }
              ↓
         All /ws/feed clients receive the message
              ↓
         Dashboard re-fetches GET /detections/ automatically
```

---

## WebSocket — Jetson Integration

The Jetson device connects to `ws://<your-server>/ws/detections` and sends detection reports as JSON.

The Jetson is responsible for:
1. Running the YOLO and acoustic detection models
2. Calling an external LLM API to generate a description
3. Uploading the captured image directly to Supabase Storage
4. Sending the detection report (including the image file path) to this backend via WebSocket

### Expected JSON format from Jetson:
```json
{
  "base_station_id": "uuid-of-the-base-station",
  "drone_detected": true,
  "yolo_confidence": 0.94,
  "acoustic_confidence": 0.87,
  "description": "LLM-generated description of the detection",
  "image_url": "folder/image-filename.jpg",
  "detected_at": "2025-06-15T14:32:00Z"
}
```

> `image_url` should be the file path within the Supabase Storage bucket, not a full URL.

### Backend response on success:
```json
{
  "status": "ok",
  "detection_id": "uuid-of-saved-detection"
}
```

### Backend response on error:
```json
{
  "status": "error",
  "detail": "error message"
}
```

---

## Authentication

The API uses JWT (JSON Web Token) bearer authentication.

1. Register a user via `POST /auth/register`
2. Login via `POST /auth/login` — returns an `access_token`
3. Include the token in the `Authorization` header for all protected requests:
```
Authorization: Bearer <your-token>
```

- Tokens expire after 30 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES` in `.env`)
- Passwords are hashed using **Argon2id** via `argon2-cffi`
- The `/auth/token` endpoint accepts form data and is used by the Swagger UI **Authorize** button

---

## Image Storage — Private Bucket with Signed URLs

Images are stored in a **private** Supabase Storage bucket. The stored value in the `image_url` column is the file path within the bucket (e.g. `detections/image.jpg`), not a publicly accessible URL.

When the frontend fetches detections, the backend calls the Supabase Storage API to generate a **signed URL** — a temporary, authenticated URL that expires after a configurable duration. This ensures images are never publicly accessible.

- Signed URL expiry is controlled by `SIGNED_URL_EXPIRY_SECONDS` in `.env` (default: 3600 seconds / 1 hour)
- Signed URL generation is handled by `app/services/storage.py`

---

## Environment Variables

Create a `.env` file in the project root. Use `.env.example` as a template:

```
DATABASE_URL=postgresql://user:password@host:5432/dbname
SECRET_KEY=your-secret-key-here
SUPABASE_URL=your-supabase-project-url
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_BUCKET=your-bucket-name
SIGNED_URL_EXPIRY_SECONDS=3600
```

| Variable | Description |
|---|---|
| `DATABASE_URL` | Supabase PostgreSQL connection string |
| `SECRET_KEY` | Secret used to sign JWT tokens (use a long random string) |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase anon/service key |
| `SUPABASE_BUCKET` | Supabase Storage bucket name for images |
| `SIGNED_URL_EXPIRY_SECONDS` | How long signed image URLs stay valid (default: 3600) |

---

## Local Development Setup

**1. Clone the repository:**
```bash
git clone <repo-url>
cd Capstone-Backend
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# Windows CMD
venv\Scripts\activate.bat
# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**
```bash
cp .env.example .env
# Fill in your actual values in .env
```

**5. Run database migrations:**
```bash
alembic upgrade head
```

**6. Start the server:**
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`
Interactive API docs at `http://localhost:8000/docs`

---

## Docker

Docker setup is handled separately by a team member. The application is Docker-ready:
- All configuration is via environment variables (`.env`)
- No hardcoded config in the codebase
- `requirements.txt` is up to date for building the image
