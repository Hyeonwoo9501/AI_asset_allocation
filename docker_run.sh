#!/bin/bash

# Docker run script for AI Asset Allocation project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}AI Asset Allocation - Docker Runner${NC}"
echo -e "${GREEN}=====================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker first. See DOCKER_INSTALL.md for instructions."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed.${NC}"
    echo "Please install docker-compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found.${NC}"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env file and add your FRED_API_KEY${NC}"
fi

# Parse command line arguments
MODE=${1:-cpu}  # Default to CPU mode

case $MODE in
    cpu)
        echo -e "${GREEN}Starting CPU container...${NC}"
        docker-compose up -d transformer-cpu
        echo -e "${GREEN}Container started. Entering shell...${NC}"
        docker-compose exec transformer-cpu bash
        ;;
    gpu)
        echo -e "${GREEN}Starting GPU container...${NC}"
        docker-compose up -d transformer-gpu
        echo -e "${GREEN}Container started. Entering shell...${NC}"
        docker-compose exec transformer-gpu bash
        ;;
    jupyter)
        echo -e "${GREEN}Starting Jupyter notebook...${NC}"
        docker-compose up -d jupyter
        echo -e "${GREEN}Jupyter started at http://localhost:8888${NC}"
        ;;
    tensorboard)
        echo -e "${GREEN}Starting TensorBoard...${NC}"
        docker-compose up -d tensorboard
        echo -e "${GREEN}TensorBoard started at http://localhost:6006${NC}"
        ;;
    build)
        echo -e "${GREEN}Building Docker images...${NC}"
        docker-compose build
        echo -e "${GREEN}Build complete.${NC}"
        ;;
    stop)
        echo -e "${GREEN}Stopping all containers...${NC}"
        docker-compose down
        echo -e "${GREEN}All containers stopped.${NC}"
        ;;
    clean)
        echo -e "${YELLOW}Removing all containers and images...${NC}"
        docker-compose down --rmi all -v
        echo -e "${GREEN}Cleanup complete.${NC}"
        ;;
    train)
        echo -e "${GREEN}Starting training in CPU container...${NC}"
        docker-compose run --rm transformer-cpu python train.py --config configs/config.yaml
        ;;
    test)
        echo -e "${GREEN}Running example script...${NC}"
        docker-compose run --rm transformer-cpu python run_example.py
        ;;
    *)
        echo "Usage: $0 {cpu|gpu|jupyter|tensorboard|build|stop|clean|train|test}"
        echo ""
        echo "Commands:"
        echo "  cpu         - Start CPU container and enter shell"
        echo "  gpu         - Start GPU container and enter shell"
        echo "  jupyter     - Start Jupyter notebook server"
        echo "  tensorboard - Start TensorBoard server"
        echo "  build       - Build Docker images"
        echo "  stop        - Stop all containers"
        echo "  clean       - Remove all containers and images"
        echo "  train       - Run training script"
        echo "  test        - Run example script"
        exit 1
        ;;
esac
