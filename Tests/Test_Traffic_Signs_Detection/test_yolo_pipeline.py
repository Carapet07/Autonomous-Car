from ultralytics import YOLO

def test_yolo_sanity():
    """
    This test runs a single tiny training loop to make sure
    YOLO + data + config all connect correctly.
    """
    
    model = YOLO('../../Traffic_Signs_Detection/yolov8n.pt')
    
    results = model.train(data="../../Datasets/TrafficSignsSet/car/data.yaml",
                         epochs=1,
                         imgsz=416,
                         batch=1, 
                         )
    
    assert results is not None
    
    preds = model.predict(source='../../Datasets/TrafficSignsSet/car/valid/images/00000_00000_00002_png.rf.109f031ac8e60eba952da43b054389c0.jpg')
    assert preds is not None

if __name__ == "__main__":
    test_yolo_sanity()
    print("All tests passed!")
    
    