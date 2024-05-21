import cv2
import numpy as np
from shapely.geometry import Polygon

def fit_circle_to_contour(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius

def calculate_circularity(polygon):
    area = polygon.area
    perimeter = polygon.length
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)

def process_image_for_contours(image_path:str):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return
    
    image_copy = image.copy()
    image_2 = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('',thresh)
    cv2.imshow('edges',edges)
    cv2.imshow('original',image_copy)
    cv2.imshow('gray',gray)
    circle_info = []
    min_area = 5000
    score = 3
    open = 0
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            center, radius = fit_circle_to_contour(contour)
            contour_area = cv2.contourArea(contour)
            circle_area = np.pi * (radius ** 2)
            roundness = contour_area / circle_area if circle_area > 0 else 0

            cv2.drawContours(image, [contour], -1, (0, 255, 0))  
            cv2.circle(image, center, radius, (255, 0, 0), 2) 

            polygon = Polygon([tuple(point[0]) for point in contour])
            convex_hull = polygon.convex_hull
            pts = np.array(list(polygon.exterior.coords), np.int32)
            pts = pts.reshape((-1, 1, 2))       
            cv2.polylines(image_2, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # Mostrar la imagen
            cv2.imshow('Polygon', image_2)


            circularity = calculate_circularity(convex_hull)
            if roundness>=0.65:
                score = 2
                closed = 0
            elif roundness< 0.65 and roundness>=0.30:
                score = 1
                closed = 0
            elif roundness < 0.30:
                score = 0

            if score == 0:
                closed = 0
                kernel = np.ones((19,19), np.uint8)
                gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                blurred = cv2.GaussianBlur(gray, (7, 7), 0)
                edges = cv2.Canny(blurred, 50, 150)
                thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 15   , 2)
                closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                contours_closing, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, 
                                                       cv2.CHAIN_APPROX_SIMPLE)

                if contours_closing:
                    contour_closing = max(contours_closing, key=cv2.contourArea)
                    center, radius = fit_circle_to_contour(contour_closing)
                    cv2.drawContours(image_2, [contour_closing], -1, (0, 255, 0), 
                                     thickness=cv2.FILLED)  
                    cv2.circle(image_2, center, radius, (255, 255, 0), 2) 
                    contour_area = cv2.contourArea(contour_closing)
                    circle_area = np.pi * (radius ** 2)
                    roundness = contour_area / circle_area if circle_area > 0 else 0
                    if roundness >= 0.30: 
                        score = 1 
                        open = 1

            circle_info.append({
                "center": center,
                "radius": radius,
                "circularity": circularity,
                "roundness": roundness,
                "score":score,
                "Closed":closed,
            })

    if not circle_info:
        print("No contours found")

    else:
        cv2.imshow('Processed Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return circle_info

# Example of use:
contour_info = process_image_for_contours('IC09219P20E20PD00011_2211140001-1.png')
print(contour_info)
