# Multimodal COVID-19 Detection Framework

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd multimodal-covid-detection

# Run setup script
chmod +x setup.sh
./setup.sh

# Start the framework
docker-compose up multimodal-covid

# For Jupyter notebook
docker-compose up jupyter
# Then visit http://localhost:8888
```

### Option 2: Conda Environment
```bash
# Create conda environment
conda env create -f environment.yml
conda activate multimodal-covid

# Install additional dependencies
pip install -r requirements.txt

# Run the framework
python multimodal_covid_detection.py
```

### Option 3: pip Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the framework
python multimodal_covid_detection.py
```

## System Requirements

- **GPU**: NVIDIA GPU with CUDA 11.6+ (recommended)
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 10GB free space for models and data
- **OS**: Ubuntu 20.04+, Windows 10+, or macOS 10.15+

## Data Preparation

1. Download COUGHVID dataset
2. Place audio files in `data/raw/audio/`
3. Place metadata CSV in `data/raw/metadata/`
4. Run preprocessing: `python preprocess_data.py`

## Model Training

```bash
# Train all models
python train_multimodal.py --folds 5 --epochs 50

# Train specific component
python train_multimodal.py --model cnn --epochs 30
python train_multimodal.py --model vit --epochs 20
python train_multimodal.py --model xgboost
```

## Container Registry

Pull pre-built image:
```bash
docker pull your-registry/multimodal-covid:latest
```

## Hardware Benchmarks

| Component | GPU Time | CPU Time | Memory |
|-----------|----------|----------|--------|
| CNN Training | 15 min | 45 min | 4GB |
| ViT Training | 2 hours | 8 hours | 12GB |
| XGBoost | 5 min | 15 min | 2GB |
| Full Pipeline | 2.5 hours | 10 hours | 16GB |
