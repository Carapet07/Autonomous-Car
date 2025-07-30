import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def validate_paths(model_path: str, data_path: str) -> bool:
    """
    Validate that model and data files exist.
    
    Args:
        model_path: Path to the YOLO model file
        data_path: Path to the data.yaml file
        
    Returns:
        bool: True if all paths are valid, False otherwise
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return False
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return False
    
    print(f"✅ Model found: {model_path}")
    print(f"✅ Data config found: {data_path}")
    return True

def main(args):
    """
    Main training function.
    
    Args:
        args: Parsed command line arguments
    """
    print("🚀 Starting YOLO Training...")
    print(f"📊 Training Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Dataset: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Project: {args.project}")
    print(f"   Name: {args.name}")
    
    # Validate paths
    if not validate_paths(args.model, args.data):
        sys.exit(1)
    
    try:
        # Load model
        print(f"📥 Loading model from {args.model}...")
        model = YOLO(args.model)
        print("✅ Model loaded successfully!")
        
        # Start training
        print("🏋️ Starting training...")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            verbose=True,
            save=True,
            save_period=5,  # Save every 5 epochs
            plots=True,     # Generate training plots
            val=True        # Run validation
        )
        
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to: {args.project}/{args.name}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO model for traffic sign detection')
    
    # Define CLI arguments
    parser.add_argument('--model', type=str, 
                       default='yolov8n.pt',  # Fixed: use yolov8n.pt
                       help='Path to YOLO model (default: yolov8n.pt)')
    
    parser.add_argument('--data', type=str, 
                       default='Datasets/TrafficSignsSet/car/data.yaml',  # Fixed: relative to project root
                       help='Path to dataset config (default: Datasets/TrafficSignsSet/car/data.yaml)')
    
    parser.add_argument('--epochs', type=int, 
                       default=10,
                       help='Number of training epochs (default: 10)')
    
    parser.add_argument('--batch', type=int, 
                       default=8,
                       help='Batch size (default: 8)')
    
    parser.add_argument('--imgsz', type=int, 
                       default=416,
                       help='Image size (default: 416)')
    
    parser.add_argument('--project', type=str, 
                       default='models/traffic_signs_detection',
                       help='Project directory (default: models/traffic_signs_detection)')
    
    parser.add_argument('--name', type=str, 
                       default='yolov8n_traffic_signs',
                       help='Experiment name (default: yolov8n_traffic_signs)')
    
    args = parser.parse_args()
    main(args)