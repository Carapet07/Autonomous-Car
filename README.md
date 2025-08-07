# Autonomous Car Project 🚗

This project focuses on object detection for self-driving cars using YOLO models. Currently implemented is a comprehensive object detection system trained on the BDD100K dataset to detect pedestrians, vehicles, traffic lights, and traffic signs. The project is designed to be extensible for additional computer vision tasks needed for autonomous driving.

## 🚀 Current Features

### Object Detection ✅
- **Model**: YOLOv8 trained on BDD100K dataset
- **Classes**: 10 object types (pedestrians, riders, cars, trucks, buses, trains, motorcycles, bicycles, traffic lights, traffic signs)
- **Auto-device detection**: Automatically uses best available device (CUDA, MPS, or CPU)
- **Progress tracking**: Real-time training progress visualization

## 📁 Project Structure

```
Autonomous-Car-main/
├── scripts/
│   └── object_detection/           # Object detection training scripts
├── notebooks/
│   └── object_detection/           # Jupyter notebooks for data analysis
├── Datasets/
│   └── ObjectDetectionSet/         # BDD100K dataset
├── models/
│   └── object_detection/           # Trained model outputs
└── Tests/                          # Test files
```

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Carapet07/Autonomous-Car.git
   cd Autonomous-Car-main
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install ultralytics torch torchvision matplotlib pillow pyyaml tqdm seaborn
   ```

## 🚀 Usage

### Object Detection Training

**Quick test (1 epoch)**:
```bash
python scripts/object_detection/train_model.py \
    --epochs 1 --batch 1
```

**Full training**:
```bash
python scripts/object_detection/train_model.py \
    --epochs 100 --batch 16 --imgsz 640
```

**Custom parameters**:
```bash
python scripts/object_detection/train_model.py \
    --data Datasets/ObjectDetectionSet/bdd100k/yolo_format/data.yaml \
    --epochs 50 --batch 8 --device auto
```

## 📊 Dataset Information

### BDD100K Object Detection Dataset
- **Location**: `Datasets/ObjectDetectionSet/bdd100k/yolo_format/`
- **Format**: YOLO format (converted from original JSON annotations)
- **Classes**: 10 object detection classes
- **Training images**: ~1,154 images
- **Validation images**: ~10,000 images

## 📚 Notebooks

The project includes comprehensive Jupyter notebooks:

1. **Data Preparation** (`notebooks/object_detection/data_preparation.ipynb`)
   - Convert BDD100K annotations to YOLO format
   - Dataset organization and validation

2. **Data Visualization** (`notebooks/object_detection/data_visualization.ipynb`)
   - Dataset statistics and class distribution
   - Sample image visualization with annotations

3. **Model Building** (`notebooks/object_detection/model_building.ipynb`)
   - Quick model testing and validation
   - Performance evaluation


The CARLA simulator is running on my PC (with an RTX 4070 Ti Super). Using the CARLA Python API, I connect from my MacBook to the simulator, so all the heavy simulation runs on the PC, while I develop and test code from the Mac

## 📄 License

This project is open source and available under the MIT License.