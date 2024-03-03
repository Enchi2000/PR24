from ultralytics import YOLO
import cv2
import numpy as np
import torch


# Load a model
model = YOLO('/home/enchi/Documentos/best.pt')  # pretrained YOLOv8n model

# classesFile="/home/enchi/Documentos/obj.names"
# with open(classesFile,'rt') as f:
#     classes=f.read().rstrip('\n').split('\n')
#     COLORS=np.random.uniform(0,255,size=(len(classes),3))

# Run batched inference on a list of images

source = cv2.imread('/home/enchi/Documentos/PEF/test_images/IC00627P01E20PD00011_2211300001-1.png')

results = model(source, conf=0.9, iou=0.5)
xyxys = []
confidences = []
class_ids = []

for result in results:
    boxes = result.boxes.cpu().numpy()
    xyxys.extend(boxes.xyxy)  # Use extend to append multiple values
    confidences.extend(boxes.conf)
    class_ids.extend(boxes.cls)
    for i, xyxy in enumerate(xyxys):
        x1, y1, x2, y2 = map(int, xyxy)
        class_name = class_ids[i]  # Assuming class_ids contains class labels
        confidence = confidences[i]  # Assuming confidences contain confidence scores
        label = f'{class_name}: {confidence:.2f}'
        cv2.rectangle(source, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(source, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow('test', source)
cv2.waitKey(0)
cv2.destroyAllWindows()