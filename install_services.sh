#!/bin/bash
# Installs and enables the U-SCAR systemd services.
# Run once after setting up your .env file:
#   chmod +x install_services.sh
#   sudo ./install_services.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$REPO_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found."
    echo "Copy .env.jetson.example to .env and fill in your values first."
    exit 1
fi

echo "Installing U-SCAR systemd services..."

cp "$REPO_DIR/uscar-vision.service" /etc/systemd/system/uscar-vision.service
cp "$REPO_DIR/uscar-audio.service"  /etc/systemd/system/uscar-audio.service

systemctl daemon-reload

systemctl enable uscar-vision.service
systemctl enable uscar-audio.service

echo ""
echo "Services installed and enabled. They will start automatically on next boot."
echo ""
echo "To start them now without rebooting:"
echo "  sudo systemctl start uscar-vision"
echo "  sudo systemctl start uscar-audio"
echo ""
echo "To check status / logs:"
echo "  sudo systemctl status uscar-vision"
echo "  sudo systemctl status uscar-audio"
echo "  journalctl -u uscar-vision -f"
echo "  journalctl -u uscar-audio -f"
echo ""
echo "To stop and disable:"
echo "  sudo systemctl stop uscar-vision uscar-audio"
echo "  sudo systemctl disable uscar-vision uscar-audio"
