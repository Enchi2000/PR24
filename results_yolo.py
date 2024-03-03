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

results=model(source)
xyxys=[]
confidences=[]
class_ids=[]
for result in results:
    boxes=result.boxes.cpu().numpy()
    xyxys=boxes.xyxy
    for xyxy in xyxys:
        cv2.rectangle(source,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(0,255,0),1)
cv2.imshow('test',source)
cv2.waitKey(0)
cv2.destroyAllWindows()