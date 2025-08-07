import sys
import os 
from pathlib import Path
from ultralytics import YOLO
import argparse
import torch

def get_best_device() -> str:
    """
    Auto-detect the best available device for training.
    
    Returns:
        str: Best available device ('cuda:0', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("🚀 Apple Metal Performance Shaders (MPS) detected")
    else:
        device = 'cpu'
        print("💻 Using CPU for training")
    
    print(f"🔧 Selected device: {device}")
    return device


def validate_paths(model_path: str, data_path: str) -> bool:
    """
    This function checks if the paths are correct

    Args:
        model_path (str): path to the model
        data_path (str): path to the data.yaml

    Returns:
        bool: The function returns if the paths are either correct or incorrect(True/False)
    """
    print(f'Model will be loaded from: {model_path}')
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print(f"Make sure the path is correct and the file exists")
        return False
    
    print(f'Data found: {data_path}')
    return True 


def train_model(args):
    if not validate_paths(args.model, args.data):
        return sys.exit(1)
    
    # Auto-detect best device if not specified
    if args.device == 'auto':
        device = get_best_device()
    else:
        device = args.device
        print(f"🔧 Using specified device: {device}")
    
    try:
        model = YOLO(args.model)
        print(f"\n📋 Training Configuration:")
        print(f"   Model: {args.model}")
        print(f"   Dataset: {args.data}")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch Size: {args.batch}")
        print(f"   Image Size: {args.imgsz}")
        print(f"   Device: {device}")
        print(f"   Project: {args.project}")
        print(f"   Name: {args.name}")
        print(f"\n🚀 Starting training...")
        
        results = model.train(
                     epochs=args.epochs,
                     batch=args.batch,
                     data=args.data,
                     imgsz=args.imgsz,
                     device=device,
                     project=args.project,
                     name=args.name,
                     verbose=True,
                     save=True,
                     plots=True,
                     val=True)
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {args.project}/{args.name}")
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f'\n❌ Training failed with error: {str(e)}')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO model for object detection on BDD100K dataset')
    
    parser.add_argument('--model', type=str, 
                        default='yolov8n.pt', 
                        help='YOLO model (default: yolov8n.pt - will download if not found)')
    parser.add_argument('--data', type=str, 
                        default='Datasets/ObjectDetectionSet/bdd100k/yolo_format/data.yaml', 
                        help='Path to dataset config (default: Datasets/ObjectDetectionSet/bdd100k/yolo_format/data.yaml)')
    parser.add_argument('--epochs', type=int,
                        default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int,
                        default=16, help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int,
                        default=640, help='Image size (default: 640)')
    parser.add_argument('--device', type=str,
                        default='auto', help='Device to use (auto, cpu, cuda:0, mps) (default: auto)')
    parser.add_argument('--project', type=str,
                        default='models/object_detection',
                        help='Project directory (default: models/object_detection)')
    parser.add_argument('--name', type=str,
                        default='bdd100k_model',
                        help='Experiment name (default: bdd100k_model)')
    
    args = parser.parse_args()
    train_model(args)