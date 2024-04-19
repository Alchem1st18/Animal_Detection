from ultralytics import YOLO

#loading model
model = YOLO("yolov8n.yaml")

#using model
results = model.train(data="config.yaml",epochs = 50  )