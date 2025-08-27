# Multimodal COVID-19 Detection Framework

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd Cough-COVID-19-Detection-

# Run setup script
chmod +x setup.sh
./setup.sh

# Start the framework
docker-compose up multimodal-covid

# For Jupyter notebook
docker-compose up jupyter

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
python 13_CNN-ViT-XGb.py
```

## System Requirements

- **GPU**: NVIDIA GPU with CUDA 11.6+ (recommended)
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 10GB free space for models and data
- **OS**: Ubuntu 20.04+, Windows 10+, or macOS 10.15+

## Data Preparation

1. Download COUGHVID dataset {public_dataset_v3.zip}: URL: https://doi.org/10.5281/zenodo.7024894 
2. Place audio files in `Specified Location`
3. Place metadata CSV in `Specified Location`
4. Run preprocessing: `Step by step: 01_Convert audio files.py ======> into 08_extract Mel.ipynb`

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
docker pull takieddinelom/multimodal-covid:v1.0.0
```

## Hardware Benchmarks

| Component | GPU Time | CPU Time | Memory |
|-----------|----------|----------|--------|
| CNN Training | 15 min | 45 min | 4GB |
| ViT Training | 2 hours | 8 hours | 12GB |
| XGBoost | 5 min | 15 min | 2GB |
| Full Pipeline | 2.5 hours | 10 hours | 16GB |
