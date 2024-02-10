import cv2
import numpy as np
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True,help = 'path to input image')
parser.add_argument('-c', '--config', required=True,help = 'path to yolo config file')
parser.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
parser.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args=parser.parse_args()

cv2.namedWindow('Image',cv2.WINDOW_NORMAL)

image=cv2.imread(args.image)
width=image.shape[1]
height=image.shape[0]
scale=0.00392

net=cv2.dnn.readNet(args.weights,args.config)

classes=None
with open(args.classes,'r') as f:
    classes=[line.strip() for line in f.readlines()]
print(len(classes))

layer_names = net.getLayerNames()
indices=net.getUnconnectedOutLayers()

output_layers = [layer_names[int(i) - 1] for i in indices]

# Detecting objects
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_id]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        cv2.putText(image, label, (x-5, y-5), font, 1, color, 1)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



