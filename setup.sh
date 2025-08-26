# - Installation script -
#!/bin/bash

echo "Setting up Multimodal COVID-19 Detection Framework..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check NVIDIA Docker runtime
if ! docker info | grep -q nvidia; then
    echo "Warning: NVIDIA Docker runtime not detected. GPU acceleration may not work."
fi

# Create necessary directories
mkdir -p data/{raw,processed,features}
mkdir -p models/{cnn,vit,xgboost,llama}
mkdir -p logs/{training,inference}
mkdir -p results/{experiments,figures}
mkdir -p notebooks

# Set permissions
chmod +x *.sh

# Build Docker image
echo "Building Docker image..."
docker-compose build

# Download pre-trained models (if available)
echo "Setting up model directories..."
# wget -O models/vit/audioset_pretrained.pth https://example.com/audioset_vit.pth

echo "Setup completed!"
echo ""
echo "To run the framework:"
echo "  docker-compose up multimodal-covid"
echo ""
echo "To run Jupyter notebook:"
echo "  docker-compose up jupyter"
echo ""
echo "To run with conda environment:"
echo "  conda env create -f environment.yml"
echo "  conda activate multimodal-covid"

---
