from ultralytics import YOLO

model=YOLO('/home/enchi/Documentos/PR24/runs/detect/S501/weights/best.pt')

model.predict('/home/enchi/Documentos/PR24/test_images/IC00627P01E20PD00011_2211300001-1.png',save=True)

