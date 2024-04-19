from ultralytics import YOLO

model = YOLO('C:/Users/Srikar/PycharmProjects/object_detection/results/runs/detect/train6/weights/best.pt')

results = model(['C:/Users/Srikar/PycharmProjects/object_detection/data/images/train/0a2ea8f93b4cb30a.jpg','C:/Users/Srikar/PycharmProjects/object_detection/data/images/test/0b2a2c061ef16759.jpg','C:/Users/Srikar/PycharmProjects/object_detection/data/images/test/9e5160657896fa20.jpg'])

for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    result.show()