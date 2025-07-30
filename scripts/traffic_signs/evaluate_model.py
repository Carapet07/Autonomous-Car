import argparse
import os
import sys
import json
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml

def load_data_config(data_path: str) -> dict:
    """
    Load and parse the data.yaml configuration file.
    
    Args:
        data_path: Path to data.yaml file
        
    Returns:
        dict: Parsed YAML configuration
    """
    try:
        with open(data_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading data config: {str(e)}")
        return None

def validate_paths(model_path: str, data_path: str, test_images_path: str = None) -> bool:
    """
    Validate that required files exist.
    
    Args:
        model_path: Path to the trained YOLO model
        data_path: Path to the data.yaml file
        test_images_path: Optional path to test images
        
    Returns:
        bool: True if all paths are valid
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return False
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return False
    
    if test_images_path and not os.path.exists(test_images_path):
        print(f"❌ Error: Test images path not found at {test_images_path}")
        return False
    
    print(f"✅ Model found: {model_path}")
    print(f"✅ Data config found: {data_path}")
    if test_images_path:
        print(f"✅ Test images found: {test_images_path}")
    return True

def evaluate_on_dataset(model, data_path: str, conf_threshold: float = 0.25):
    """
    Evaluate model on the validation dataset.
    
    Args:
        model: Loaded YOLO model
        data_path: Path to data.yaml file
        conf_threshold: Confidence threshold for predictions
        
    Returns:
        dict: Evaluation results
    """
    print("🔍 Evaluating model on validation dataset...")
    
    try:
        # Run validation
        results = model.val(
            data=data_path,
            conf=conf_threshold,
            verbose=True,
            save_json=True,
            save_txt=True
        )
        
        print("✅ Validation completed!")
        return results
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return None

def predict_on_single_image(model, image_path: str, conf_threshold: float = 0.25, save_results: bool = True):
    """
    Run prediction on a single image.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        conf_threshold: Confidence threshold
        save_results: Whether to save prediction results
        
    Returns:
        list: Prediction results
    """
    print(f"🖼️ Running prediction on {image_path}...")
    
    try:
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            save=save_results,
            save_txt=save_results,
            verbose=True
        )
        
        print("✅ Prediction completed!")
        return results
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return None

def predict_on_multiple_images(model, images_dir: str, conf_threshold: float = 0.25):
    """
    Run predictions on multiple images in a directory.
    
    Args:
        model: Loaded YOLO model
        images_dir: Directory containing images
        conf_threshold: Confidence threshold
        
    Returns:
        list: All prediction results
    """
    print(f"📁 Running predictions on images in {images_dir}...")
    
    try:
        results = model.predict(
            source=images_dir,
            conf=conf_threshold,
            save=True,
            save_txt=True,
            verbose=True
        )
        
        print(f"✅ Predictions completed on {len(results)} images!")
        return results
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {str(e)}")
        return None

def print_evaluation_summary(results, data_config: dict):
    """
    Print a summary of evaluation results.
    
    Args:
        results: Validation results from model.val()
        data_config: Data configuration dictionary
    """
    if not results:
        print("❌ No results to summarize")
        return
    
    print("\n" + "="*50)
    print("📊 EVALUATION SUMMARY")
    print("="*50)
    
    # Print class names if available
    if data_config and 'names' in data_config:
        print(f"🎯 Classes: {len(data_config['names'])}")
        for i, name in enumerate(data_config['names']):
            print(f"   {i}: {name}")
    
    # Print metrics if available
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n📈 Metrics:")
        print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"   Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"   Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    
    print("="*50)

def main(args):
    """
    Main evaluation function.
    
    Args:
        args: Parsed command line arguments
    """
    print("🔍 Starting YOLO Model Evaluation...")
    print(f"📊 Evaluation Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Data Config: {args.data}")
    print(f"   Confidence Threshold: {args.conf}")
    print(f"   Test Images: {args.test_images}")
    print(f"   Output Directory: {args.output_dir}")
    
    # Validate paths
    if not validate_paths(args.model, args.data, args.test_images):
        sys.exit(1)
    
    try:
        # Load model
        print(f"📥 Loading model from {args.model}...")
        model = YOLO(args.model)
        print("✅ Model loaded successfully!")
        
        # Load data configuration
        data_config = load_data_config(args.data)
        
        # Run validation on dataset
        if args.validate:
            val_results = evaluate_on_dataset(model, args.data, args.conf)
            print_evaluation_summary(val_results, data_config)
        
        # Run predictions on test images
        if args.test_images:
            if os.path.isdir(args.test_images):
                # Directory of images
                pred_results = predict_on_multiple_images(model, args.test_images, args.conf)
            else:
                # Single image
                pred_results = predict_on_single_image(model, args.test_images, args.conf, args.save_results)
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate YOLO model for traffic sign detection')
    
    # Define CLI arguments
    parser.add_argument('--model', type=str, 
                       default='models/traffic_signs_detection/yolov8n_traffic_signs/weights/best.pt',
                       help='Path to trained YOLO model (default: models/traffic_signs_detection/yolov8n_traffic_signs/weights/best.pt)')
    
    parser.add_argument('--data', type=str, 
                       default='Datasets/TrafficSignsSet/car/data.yaml',
                       help='Path to dataset config (default: Datasets/TrafficSignsSet/car/data.yaml)')
    
    parser.add_argument('--test-images', type=str, 
                       default=None,
                       help='Path to test image or directory of images')
    
    parser.add_argument('--conf', type=float, 
                       default=0.25,
                       help='Confidence threshold (default: 0.25)')
    
    parser.add_argument('--output-dir', type=str, 
                       default='evaluation_results',
                       help='Output directory for results (default: evaluation_results)')
    
    parser.add_argument('--validate', action='store_true',
                       help='Run validation on the dataset')
    
    parser.add_argument('--save-results', action='store_true',
                       help='Save prediction results')
    
    args = parser.parse_args()
    main(args)
