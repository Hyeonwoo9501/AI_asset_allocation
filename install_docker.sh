#!/bin/bash

# Docker Installation Script for WSL2 Ubuntu
# This script installs Docker Engine on Ubuntu in WSL2

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Docker Installation Script for WSL2${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running on Ubuntu
if ! grep -q "Ubuntu" /etc/os-release; then
    echo -e "${RED}Error: This script is designed for Ubuntu.${NC}"
    exit 1
fi

# Check if Docker is already installed
if command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker is already installed!${NC}"
    docker --version
    echo ""
    read -p "Do you want to reinstall? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Skipping installation.${NC}"
        exit 0
    fi
fi

echo -e "${BLUE}[1/7] Removing old Docker versions...${NC}"
sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

echo -e "${BLUE}[2/7] Installing required packages...${NC}"
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

echo -e "${BLUE}[3/7] Adding Docker GPG key...${NC}"
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo -e "${BLUE}[4/7] Setting up Docker repository...${NC}"
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo -e "${BLUE}[5/7] Installing Docker Engine...${NC}"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo -e "${BLUE}[6/7] Starting Docker service...${NC}"
sudo service docker start

echo -e "${BLUE}[7/7] Adding current user to docker group...${NC}"
sudo usermod -aG docker $USER

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Docker installation completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Important: You need to log out and log back in (or restart WSL2) for group changes to take effect.${NC}"
echo ""
echo "To restart WSL2, run this in Windows PowerShell:"
echo -e "${BLUE}  wsl --shutdown${NC}"
echo ""
echo "After restarting, verify installation:"
echo -e "${BLUE}  docker --version${NC}"
echo -e "${BLUE}  docker run hello-world${NC}"
echo ""

# Check if systemd is enabled
if grep -q "systemd=true" /etc/wsl.conf 2>/dev/null; then
    echo -e "${GREEN}systemd is enabled. Docker will start automatically.${NC}"
else
    echo -e "${YELLOW}systemd is not enabled. You'll need to start Docker manually each time:${NC}"
    echo -e "${BLUE}  sudo service docker start${NC}"
    echo ""
    echo "To enable systemd (requires WSL2 restart):"
    echo -e "${BLUE}  sudo bash -c 'cat > /etc/wsl.conf << EOF${NC}"
    echo -e "${BLUE}[boot]${NC}"
    echo -e "${BLUE}systemd=true${NC}"
    echo -e "${BLUE}EOF'${NC}"
    echo -e "${BLUE}  # Then restart WSL2 from PowerShell: wsl --shutdown${NC}"
fi

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Restart WSL2"
echo "2. Verify Docker: docker --version"
echo "3. Test Docker: docker run hello-world"
echo "4. Build project: cd /mnt/c/projects/AI_asset_allocation && bash docker_run.sh build"
echo "5. Run project: bash docker_run.sh cpu"
