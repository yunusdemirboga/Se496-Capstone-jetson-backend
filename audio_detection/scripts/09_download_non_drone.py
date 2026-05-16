#!/usr/bin/env python3
"""
Script 09: Download targeted non-drone hard-negative audio clips.

Downloads music and speech clips to widen the non-drone training class.
These are the categories causing real-world false positives on phone audio.

Uses yt-dlp with node.js runtime (no ffmpeg required).
Audio is downloaded in native format; script 10 handles slicing into windows.

Sources:
  - Music: CC-licensed / YouTube Audio Library instrumental tracks
  - Speech: CC-licensed educational/documentary speech content

Usage:
    python scripts/09_download_non_drone.py
    python scripts/09_download_non_drone.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

OUT_ROOT = Path("data/raw_non_drone")
NODE_PATH = "/usr/local/bin/node"

# YT-DLP base flags — node runtime, native audio format (no ffmpeg needed)
YTDLP_BASE = [
    "yt-dlp",
    "--no-js-runtimes",
    f"--js-runtimes", f"node:{NODE_PATH}",
    "-f", "bestaudio",          # best quality audio, native container
    "--no-playlist",
    "--quiet",
    "--no-warnings",
]

# ---------------------------------------------------------------------------
# Music: YouTube Audio Library (free, no copyright restrictions)
# Instrumental tracks covering electronic, hip-hop, ambient, pop
# ---------------------------------------------------------------------------
MUSIC_CLIPS = [
    # Electronic / bass-heavy — hardest negatives (freq overlap with drones)
    ("electronic_ncs_01",       "TW9d8vYrVFQ"),  # Elektronomia - Sky High (progressive house)
    ("electronic_ncs_02",       "__CRWE-L45k"),  # Electro-Light - Symbolism (trap, bass)
    # Hip-hop beats — rhythmic, mid-frequency content
    ("hiphop_lofi_01",          "n61ULEU7CO0"),  # lofi hip hop beats mix
    ("hiphop_lofi_02",          "CLeZyIID9Bo"),  # chill lofi mix
    # Ambient / soft — easy negatives to anchor the class
    ("ambient_01",              "DRFHklnN-SM"),  # tranquility ambient
    ("ambient_02",              "H4BAEf5V-Yc"),  # ethereal ambient instrumental
]

# ---------------------------------------------------------------------------
# Speech: CC-licensed educational and documentary content on YouTube
# Covers male/female voices, varied pace, different acoustic environments
# ---------------------------------------------------------------------------
SPEECH_CLIPS = [
    # TED-Ed educational — clear speech, studio quality
    ("speech_educational_01",   "arj7oStGLkU"),  # TED-Ed lesson
    ("speech_educational_02",   "NbuUW9i-mHs"),  # TED-Ed science
    # Documentary narration — different acoustic feel
    ("speech_documentary_01",   "0fKBhvDjuy0"),  # nature/documentary style
    # Interview / conversational — closer to phone audio
    ("speech_interview_01",     "H14bBuluwB8"),  # interview style
    # Lecture / slower cadence
    ("speech_lecture_01",       "zHL9GP_B30E"),  # educational lecture
]


def download_clip(stem: str, yt_id: str, out_dir: Path, dry_run: bool) -> bool:
    """Download a single YouTube clip in native audio format."""
    # yt-dlp output template — will produce e.g. stem.webm or stem.m4a
    out_template = str(out_dir / f"{stem}.%(ext)s")

    # Check if already downloaded (any extension)
    existing = list(out_dir.glob(f"{stem}.*"))
    if existing:
        print(f"  [skip] {stem} already exists ({existing[0].suffix})")
        return True

    if dry_run:
        print(f"  [dry-run] yt:{yt_id} → {stem}.*")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = YTDLP_BASE + [
        "--output", out_template,
        f"https://www.youtube.com/watch?v={yt_id}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout).strip()[:150]
        print(f"  [fail] {stem} — {err}")
        return False

    downloaded = list(out_dir.glob(f"{stem}.*"))
    if downloaded:
        size_kb = downloaded[0].stat().st_size // 1024
        print(f"  [ok]   {stem}{downloaded[0].suffix} ({size_kb} KB)")
        return True

    print(f"  [fail] {stem} — file not found after download")
    return False


def main(args: argparse.Namespace) -> None:
    project_root = Path(__file__).parent.parent
    music_dir  = project_root / OUT_ROOT / "music"
    speech_dir = project_root / OUT_ROOT / "speech"

    print("=" * 60)
    print("UAV Non-Drone Clip Downloader")
    print("=" * 60)
    print(f"Node runtime: {NODE_PATH}")

    # ---- Music ----
    print(f"\n[1/2] Music → {music_dir}")
    music_ok = sum(
        download_clip(stem, yt_id, music_dir, args.dry_run)
        for stem, yt_id in MUSIC_CLIPS
    )

    # ---- Speech ----
    print(f"\n[2/2] Speech → {speech_dir}")
    speech_ok = sum(
        download_clip(stem, yt_id, speech_dir, args.dry_run)
        for stem, yt_id in SPEECH_CLIPS
    )

    print("\n" + "=" * 60)
    print(f"Music:  {music_ok}/{len(MUSIC_CLIPS)} clips")
    print(f"Speech: {speech_ok}/{len(SPEECH_CLIPS)} clips")
    if not args.dry_run:
        print("\nNEXT STEP: python scripts/10_build_non_drone_set.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files.")
    main(parser.parse_args())
