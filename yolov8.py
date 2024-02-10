from ultralytics import YOLO

model=YOLO('/home/enchi/Documentos/PEF/runs/detect/small/weights/best.pt')

model.predict('/home/enchi/Documentos/PEF/test_images/IC00627P01E20PD00011_2211300001-1.png',save=True)

