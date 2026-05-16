"""
test_backend.py — Verify the U-SCAR backend connection end-to-end.

Runs four checks in order:
  1. HTTP health check  (GET /)
  2. Auth              (POST /auth/login)
  3. Base stations     (GET /base_stations/)
  4. WebSocket send    (send a mock detection to /ws/detections, check ack)

Usage:
    python3 test_backend.py --host 192.168.1.50 --username admin --password secret

    # Or read host from .env (BACKEND_WS_URL):
    python3 test_backend.py --username admin --password secret
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="U-SCAR backend connection test")
    p.add_argument("--host",     default=None,
                   help="Backend host:port, e.g. 192.168.1.50:8000 (auto-detected from .env if omitted)")
    p.add_argument("--username", required=True, help="Backend username")
    p.add_argument("--password", required=True, help="Backend password")
    return p.parse_args()


def resolve_host(args) -> str:
    if args.host:
        return args.host.rstrip("/")
    ws_url = os.getenv("BACKEND_WS_URL", "")
    if ws_url:
        # ws://192.168.1.50:8000/ws/detections  →  192.168.1.50:8000
        # wss://myapp.railway.app/ws/detections →  myapp.railway.app
        host = ws_url.replace("wss://", "").replace("ws://", "").split("/")[0]
        return host
    sys.exit("ERROR: Provide --host or set BACKEND_WS_URL in .env")


def check(label: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    line   = f"  [{status}] {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    return ok


async def run(args):
    import requests
    import websockets

    host     = resolve_host(args)
    # Preserve https/wss for remote deployments (e.g. Railway)
    ws_env   = os.getenv("BACKEND_WS_URL", "")
    secure   = ws_env.startswith("wss://")
    http_url = f"{'https' if secure else 'http'}://{host}"
    ws_url   = f"{'wss' if secure else 'ws'}://{host}/ws/detections"

    print(f"\nBackend: {http_url}")
    print("=" * 50)

    all_ok = True

    # ── 1. Health check ───────────────────────────────────────────────────────
    print("\n1. HTTP health check")
    try:
        r = requests.get(f"{http_url}/", timeout=5)
        ok = r.status_code == 200
        all_ok &= check("GET /", ok, r.json().get("message", "") if ok else f"HTTP {r.status_code}")
    except Exception as e:
        all_ok &= check("GET /", False, str(e))

    # ── 2. Auth ───────────────────────────────────────────────────────────────
    print("\n2. Auth")
    token = None
    try:
        r = requests.post(
            f"{http_url}/auth/login",
            json={"username": args.username, "password": args.password},
            timeout=5,
        )
        ok = r.status_code == 200
        if ok:
            token = r.json().get("access_token")
        all_ok &= check("POST /auth/login", ok,
                        "token received" if ok else f"HTTP {r.status_code}: {r.text}")
    except Exception as e:
        all_ok &= check("POST /auth/login", False, str(e))

    # ── 3. Base stations ──────────────────────────────────────────────────────
    print("\n3. Base stations")
    station_id = None
    if token:
        try:
            r = requests.get(
                f"{http_url}/base_stations/",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5,
            )
            ok = r.status_code == 200
            stations = r.json() if ok else []
            detail   = f"{len(stations)} station(s) found" if ok else f"HTTP {r.status_code}"
            all_ok  &= check("GET /base_stations/", ok, detail)
            if ok and stations:
                station_id = stations[0]["id"]
                print(f"         Using station: {stations[0]['name']}  ({station_id})")
            elif ok:
                print("         No base stations found — create one first via POST /base_stations/")
        except Exception as e:
            all_ok &= check("GET /base_stations/", False, str(e))
    else:
        check("GET /base_stations/", False, "skipped (no token)")
        all_ok = False

    # ── 4. WebSocket mock detection ───────────────────────────────────────────
    print("\n4. WebSocket detection ingestion")
    if station_id:
        payload = {
            "base_station_id":     station_id,
            "drone_detected":      True,
            "yolo_confidence":     0.95,
            "acoustic_confidence": None,
            "image_url":           None,
            "description":         "[test] Automated connection test — not a real detection.",
            "detected_at":         datetime.now(timezone.utc).isoformat(),
        }
        try:
            async with websockets.connect(ws_url, open_timeout=5) as ws:
                await ws.send(json.dumps(payload))
                ack = await asyncio.wait_for(ws.recv(), timeout=5.0)
                ack_data = json.loads(ack)
                ok = ack_data.get("status") == "ok"
                all_ok &= check(
                    "WS /ws/detections send + ack", ok,
                    f"detection_id={ack_data.get('detection_id', '?')}" if ok
                    else str(ack_data),
                )
        except Exception as e:
            all_ok &= check("WS /ws/detections send + ack", False, str(e))
    else:
        check("WS /ws/detections send + ack", False,
              "skipped — need at least one base station")
        all_ok = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if all_ok:
        print("All checks passed. Backend is ready.\n")
        if station_id:
            print(f"Set this in your .env:\n  BASE_STATION_ID={station_id}\n")
    else:
        print("One or more checks failed. Fix the issues above before running pipeline.py.\n")

    return all_ok


def main():
    args = parse_args()
    ok   = asyncio.run(run(args))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
