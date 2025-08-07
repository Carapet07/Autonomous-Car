from pathlib import Path

folder_path = Path('Datasets/ObjectDetectionSet/coco128/images/train2017')

if folder_path.exists():
    file_count = sum(1 for f in folder_path.iterdir() if f.is_file())
    print(f"📁 Found {file_count} files in {folder_path}")
else:
    print(f"❌ Folder not found: {folder_path}")
    
