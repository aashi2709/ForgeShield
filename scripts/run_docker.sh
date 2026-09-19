#!/bin/sh

set -eu

echo "======================================"
echo "        ForgeShield Deployment"
echo "======================================"

echo ""
echo "[1/2] Building ForgeShield..."
docker compose build

echo ""
echo "[2/2] Starting ForgeShield..."
docker compose up -d

echo ""
echo "======================================"
echo " ForgeShield is now running"
echo " http://localhost:8501"
echo "======================================"