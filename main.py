# ------------------------ Importar Librerias ----------------------- #
import runpy
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import shutil
import math
from contour import process_image_for_contours
from yolov5 import detect
import pandas
#Graficamos todos lo números encontrados que se guerdan en la carpeta
def display(folder,img1,img2,img3,score_manecillas,score_numeros,score_contour,puntuacion_final):
    # Define la ruta a la carpeta donde están almacenadas tus imágenes
    folder_path = folder
    
    # Lista para guardar las rutas de las imágenes
    image_paths = []

    # Recorre los archivos en la carpeta y agrega las imágenes a la lista
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Asegúrate de incluir los formatos que necesitas
            image_paths.append(os.path.join(folder_path, filename))

    # Determina cuántas imágenes hay
    num_images = len(image_paths)

    # Cálculo del número de filas y columnas para la visualización
    cols = 6  # Número máximo de columnas
    rows = (num_images + cols - 1) // cols  # Calcula las filas necesarias

    # Crea una figura para mostrar las imágenes
    fig, axs = plt.subplots(rows, cols, figsize=(8, 8))
    fig.tight_layout()

    # Asegurarse de que axs sea siempre un array bidimensional
    if num_images <= cols:
        axs = axs[np.newaxis, :]  # Añade una dimensión de fila si solo hay una fila

    # Si tienes menos subplots que imágenes, oculta los axes adicionales
    for ax in axs.flatten():
        ax.axis('off')
        
    # Muestra cada imagen
    for i, img_path in enumerate(image_paths):
        img = mpimg.imread(img_path)
        ax = axs[i // cols, i % cols]
        ax.imshow(img)
        ax.axis('on')  # Muestra el eje si es necesario
 
    fig2, axs2 = plt.subplots(1, 3)  # Crea una figura y una matriz de subplots (2x2)

    img1_rgb = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    img3_rgb = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
    
    # Mostrar cada imagen en su respectivo subplot
    axs2[0].imshow(img1_rgb, cmap='gray')
    axs2[0].set_title('Números Detectados')
    axs2[0].set_xlabel('Puntuacion ' + str(score_numeros))
    
    
    axs2[1].imshow(img2_rgb, cmap='gray')
    axs2[1].set_title('Manecillas Detectadas')
    axs2[1].set_xlabel('Puntuacion ' + str(score_manecillas))  # Título para el eje x del tercer subplot

    axs2[2].imshow(img3_rgb, cmap='gray')
    axs2[2].set_title('Contorno')
    axs2[2].set_xlabel('Puntuacion ' + str(score_contour))

    # Ajusta el layout para evitar que los títulos se solapen
    fig2.tight_layout()
    
    #Agregar un título general a la figura
    fig2.suptitle(f'Análisis Completo de la imagen. Evaluacion {puntuacion_final}')
    
    fig2.savefig('pruebas/mc5-1.png')
    
    plt.show()
    
def vaciar_carpeta(ruta_carpeta):
    # Comprobar si la ruta existe y es un directorio
    if not os.path.isdir(ruta_carpeta):
        print(f"La ruta especificada {ruta_carpeta} no es un directorio o no existe.")
        return

    # Listar todos los archivos y subdirectorios en el directorio
    for nombre in os.listdir(ruta_carpeta):
        # Construir ruta completa
        ruta_completa = os.path.join(ruta_carpeta, nombre)

        try:
            # Verificar si es un archivo o directorio y eliminarlo
            if os.path.isfile(ruta_completa) or os.path.islink(ruta_completa):
                os.unlink(ruta_completa)  # Eliminar archivos o enlaces simbólicos
            elif os.path.isdir(ruta_completa):
                shutil.rmtree(ruta_completa)  # Eliminar subdirectorios y su contenido
        except Exception as e:
            print(f"Error al eliminar {ruta_completa}. Razón: {e}")



#Ejecutamos un script por su ruta
dibujar = runpy.run_path('draw.py')
manecillas = runpy.run_path('contour_segmentation3.py')
numeros = runpy.run_path('results_yolo.py')

clock_results = detect.run(weights='yolov5s.pt',source = 'test_images/original_image.png' )

# Check if detections are available
if clock_results is not None:
    # Print clock_results
    print(clock_results.pandas().xyxy[0])  # Access detections as DataFrame
else:
    print("No objects detected.")

# -------------------------- Initialization ---------------------- #
imagen_para_numeros = numeros['draw']
imagen_para_angulos = imagen_para_numeros.copy()
imagen_para_manecillas = manecillas['img']
imagen_de_contorno = manecillas['final_contour']

# -------------------------- Angles and lines ---------------------- #

angulo_entre_manecillas = manecillas['angulo_entre_manecillas']
lineas_manecillas = manecillas['line_lenghts']
angulo_manecillas_2pm = manecillas['clockhand_angle_2']
angulo_manecillas_11am = manecillas['clockhand_angle_11']
lowest_point = manecillas['lowest_point']
cantidad_manecillas = len(lineas_manecillas)
#Detectamos angulos
angulos = manecillas['grouped_angles']

#Si las manecillas son dos
if cantidad_manecillas == 2:
    try:    
        angulo_manecillas_2pm = angulos[0][0]
    except IndexError:
        angulo_manecillas_2pm = 0
    try:
        angulo_manecillas_11am = angulos[1][0]
    except IndexError:
        angulo_manecillas_11am = 0
        
    angulo_entre_manecillas = angulo_manecillas_11am-angulo_manecillas_2pm

    
    #Si el gap entre ambas es maor que 40 grad
    if angulo_entre_manecillas>30:
        try:
            longitud_manecilla_2pm = angulos[0][1]
            longitud_manecilla_11am = angulos[1][1]
            diferencia_manecillas_porcentual = (longitud_manecilla_2pm - longitud_manecilla_11am) / longitud_manecilla_2pm 
            diferencia_manecillas_bool = True if diferencia_manecillas_porcentual > 0.25 else False
            print(diferencia_manecillas_porcentual) 
        except IndexError:
            longitud_manecilla_2pm  = 0
            longitud_manecilla_11am = 0
            diferencia_manecillas_porcentual = 0
            diferencia_manecillas_bool = 0
            print("No se encontraron manecillas")
            flag = 0
    #Si el gap
    else:
        try:
            #Longitud de manecilla y angulo se recorre porque la manecilla esta del lado izquierda
            longitud_manecilla_11am = angulos[0][1]
            longitud_manecilla_2pm = angulos[1][1]
            diferencia_manecillas_porcentual = (longitud_manecilla_2pm - longitud_manecilla_11am) / longitud_manecilla_2pm 
            diferencia_manecillas_bool = True if diferencia_manecillas_porcentual > 0.25 else False
            print(diferencia_manecillas_porcentual)
        except IndexError:
            longitud_manecilla_11am = 0
            longitud_manecilla_2pm = 0
            diferencia_manecillas_bool = 0
            diferencia_manecillas_porcentual = 0
            print("No se encontraron manecillas")
            flag = 0


if cantidad_manecillas == 3:
    try:
        angulo_manecillas_2pm = angulos[0][0]
    except IndexError:
        angulo_manecillas_2pm = 0
        
    try:
        angulo_manecillas_11am = angulos[2][0]
    except IndexError:
        angulo_manecillas_11am = 0
    
    angulo_entre_manecillas = angulo_manecillas_11am-angulo_manecillas_2pm


    #Si el gap entre ambas es maor que 40 grados
    if angulo_entre_manecillas>30:
    #Intetamos ver el error
        try:
            longitud_manecilla_2pm = angulos[0][1]
            longitud_manecilla_11am = angulos[2][1]
            diferencia_manecillas_porcentual = (longitud_manecilla_2pm - longitud_manecilla_11am) / longitud_manecilla_2pm 
            diferencia_manecillas_bool = True if diferencia_manecillas_porcentual > 0.25 else False
            print(diferencia_manecillas_porcentual)
        except IndexError:
            longitud_manecilla_11am=0
            longitud_manecilla_2pm=0
            diferencia_manecillas_porcentual=0
            diferencia_manecillas_bool=0
            print("No se encontraron manecillas")
            flag = 0
    else:
        try:
            #Longitud de manecilla y angulo se recorre porque la manecilla esta del lado izquierda
            longitud_manecilla_11am = angulos[0][1]
            longitud_manecilla_2pm = angulos[2][1]
            diferencia_manecillas_porcentual = (longitud_manecilla_2pm - longitud_manecilla_11am) / longitud_manecilla_2pm 
            diferencia_manecillas_bool = True if diferencia_manecillas_porcentual > 0.25 else False
            print(diferencia_manecillas_porcentual)
        except IndexError:
            longitud_manecilla_11am = 0
            longitud_manecilla_2pm = 0
            diferencia_manecillas_porcentual = 0
            diferencia_manecillas_bool = 0
            print("No se encontraron manecillas")
            flag = 0
# -------------------------- Números -------------------------- #
#Coordenadas detectadas por contornos por contorno 
detectado = numeros['detected']

#Obtenemos angulo para 2pm
try:
    upper_left_corner_2pm = np.rad2deg(numeros['upper_left_corner_2pm'])
    lower_right_corner_2pm = np.rad2deg(numeros['lower_right_corner_2pm'])
    coordenadas_2pm = numeros['coords_2pm']

#Si no lo detecto YOLO, usamos lo que esta segun su aproximación
except KeyError:
    
    for detected in detectado:
        #Unpack the bounding box and the lowest point
        x, y, w, h, label = detected
        lowest_x, lowest_y = lowest_point
        
        if label == 2:
            #Calculate the coordinates of the upper-right and lower-left corners
            upper_left = (x,y)
            lower_right = (x + w, y + h)

            upper_left_corner_2pm =  np.rad2deg(math.atan2((lowest_y-y),(x-lowest_x)))
            lower_right_corner_2pm = np.rad2deg(math.atan2((lowest_y-(y+h)),(x+w-lowest_x)))
            print(lower_right_corner_2pm,"<x<",upper_left_corner_2pm )
            coordenadas_2pm= (x,x+w,y,y+h)
        else:
            #Calculate the coordinates of the upper-right and lower-left corners
            upper_left = (0,0)
            lower_right = (0,0)

            upper_left_corner_2pm =  0
            lower_right_corner_2pm = 0
            
            coordenadas_2pm= (0,0,0,0)
            
#Obtenemos angulos para la hora 11
try:
    upper_right_corner_11am = np.rad2deg(numeros['upper_right_corner_11am'])
    lower_left_corner_11am = np.rad2deg(numeros['lower_left_corner_11am'])    
    coordenadas_11am = numeros['coords_11am']

#Si no lo detecta YOLO, ponemos lo que está según su posición
except KeyError:
    #Iteramos sobre lo que supuestamente detecto
    for detected in detectado:
    #Unpack the bounding box and the lowest point
        x, y, w, h, label = detected
        lowest_x, lowest_y = lowest_point
        
        if label == 11:
            #Calculate the coordinates of the upper-right and lower-left corners
            upper_right = (x + w, y)
            lower_left = (x, y + h)
    
            #Getting corner angles for 11
            upper_right_corner_11am = np.rad2deg(math.atan2(lowest_y-y,(x+w)-lowest_x))
            lower_left_corner_11am = np.rad2deg(math.atan2(lowest_y-(y+h),x-lowest_x))
            coordenadas_11am = (x,x+w,y,y+h)
            print(lower_left_corner_11am,"<x<",upper_right_corner_11am )
        else:
           #Calculate the coordinates of the upper-right and lower-left corners
            upper_right = (0,0)
            lower_left = (0,0)
    
            #Getting corner angles for 11
            upper_right_corner_11am = 0
            lower_left_corner_11am = 0
            coordenadas_11am = (0,0,0,0)
            

# Suponemos que numeros['class_names'] es una lista que contiene algunos números detectados.
numeros_detectados = numeros['class_names']
numeros_detectados_contorno = numeros['detected']
numeros_detectados_correcto = numeros['detected_correct']
numeros_detectados_con_error_espacial = numeros['detected_in_wrong_arrangement']
detectado_algo = numeros['detected_something']

#Imprimimos para ver que me esta entregando
print(f"Numeros detectados:{numeros_detectados}")
print(f"Numeros desde el contorno:{numeros_detectados_contorno}")
print(f"Numeros detectados correcto por YOLO:{numeros_detectados_correcto}")
print(f"Numeros con error espacial:{numeros_detectados_con_error_espacial}")
print(f"Detectó algo:{detectado_algo}")

#Extraemos las etiquetas
#Extrayendo el primer elemento de cada tupla en las listas
numeros_desde_contorno = [tupla[4] for tupla in numeros_detectados_contorno if isinstance(tupla, tuple)]
numeros_correctos = [tupla[0] for tupla in numeros_detectados_correcto if isinstance(tupla, tuple)]
numeros_con_error_espacial = [tupla[0] for tupla in numeros_detectados_con_error_espacial if isinstance(tupla, tuple)]
numeros_algo_detectado = [tupla[0] for tupla in detectado_algo if isinstance(tupla, tuple)]


#Imprimir los resultados
print("Números desde contorno:", numeros_desde_contorno)
print("Números detectados correctamente:", numeros_correctos)
print("Números con error espacial:", numeros_con_error_espacial)
print("Números algo detectado:", numeros_algo_detectado)

#Crear un conjunto para unificar todos los números y eliminar duplicados
numeros_unicos = set()

# Agregar todos los números detectados al conjunto
numeros_unicos.update(numeros_detectados)
numeros_unicos.update(numeros_desde_contorno)
numeros_unicos.update(numeros_correctos)
numeros_unicos.update(numeros_con_error_espacial)
numeros_unicos.update(numeros_algo_detectado)

# Convertir el conjunto a lista de enteros
numeros_final = [int(numero) for numero in numeros_unicos]

# Imprimir la lista de números convertidos a enteros
print("Números detectados (sin duplicados, convertidos a enteros):", numeros_final)

#Angulo de cada centroide del bbox
angles_got = numeros['angles_got']

#Centroide de reloj
centerx = numeros['centerx']
centery = numeros['centery']

#Circulo en medio de la imagen
cv2.circle(imagen_para_angulos,lowest_point,4,(255,0,0,),-1)
        
#1. Evaluacion de manecillas
# Hay diferencia entre la manecilla de minutos
etiquetas = []
angles = []

#Vemos que etiquetas y angulos tenemos
for label,real_angle in angles_got:
    etiquetas.append(label)
    angles.append(real_angle)
 
if len(manecillas) == 0:
    score_m  = 0   
    print(f"No se encontraron manecillas en el reloj. {score_m}")
else:
    print(f"Longitud Manecilla 2 {longitud_manecilla_2pm}")
    print(f"Longitud Manecilla 11 {longitud_manecilla_11am}")
    print(f"Angulo 2 manecillas {angulo_manecillas_2pm}")
    print(f"Angulo 11 manecillas {angulo_manecillas_11am}")
    print(f"Angulo entre manecillas {angulo_entre_manecillas}")
    print(f"Cantidad de manecillas detectadas {cantidad_manecillas}")
    score_m = 0
    
    #Si existe diferencia en las manecillas y apunta a ambas horas
    if diferencia_manecillas_bool == True and (2 in numeros_final and 11 in numeros_final):
        
        well_placed_2pm = lower_right_corner_2pm<angulo_manecillas_2pm<upper_left_corner_2pm
        well_placed_11pm = upper_right_corner_11am<angulo_manecillas_11am<lower_left_corner_11am
        
        
        #Se respeta la diferencia de longitud de manecillas y ambas apuntan a la hora que corresponde
        if well_placed_11pm == True and well_placed_2pm == True:
            score_m = 4
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas y ambas manecillas apunta a la hora")
            print(f"Hands are in correct position and the size diference is respetected {score_m}")
        
        #Se respeta la diferenicia de manecillas pero solo se apunta a hacia a una hora
        if (well_placed_11pm == True and well_placed_2pm == False) or (well_placed_11pm == False and well_placed_2pm == True):
            score_m = 3
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas pero solo una apunta hacia una hora")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
        
        #Se respeta la diferencia de longitud de manecillas pero no apunta hacia ninguna hora
        if well_placed_11pm == False and well_placed_2pm == False:
            score_m = 2
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas y no apunta hacia ninguna hora")
            print(f"Major Errors in the placement of the hands (Significantly out of course incluiding 10 to 11) {score_m}") 
            
    #Si no se respeta la diferencia de las medidas de las manecillas 
    if diferencia_manecillas_bool == False and (11 in numeros_final and 2 in numeros_final):

        well_placed_2pm = lower_right_corner_2pm<angulo_manecillas_2pm<upper_left_corner_2pm
        well_placed_11pm = upper_right_corner_11am<angulo_manecillas_11am<lower_left_corner_11am

        #No se respeta la diferencia de medidas y ambas manecillas apuntan a la hora
        if well_placed_11pm == True and well_placed_2pm == True:
            score_m = 3
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeto la diferencia de medidas de las manecillas pero apunta ambas horas")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
            
        #No se respeta la diferenicia de manecillas y solo se apunta a hacia a una hora
        if (well_placed_11pm == True and well_placed_2pm == False) or (well_placed_11pm == False and well_placed_2pm == True):
            score_m = 2
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeto la diferencia de las medidas de las manecillas y solo apunta hacia una hora")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
            
        #Se respeta la diferencia de longitud de manecillas pero no apunta hacia ninguna hora
        if well_placed_11pm == False and well_placed_2pm == False:
            score_m= 1
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeta la diferencia de las medidas de las manecillas y ninguna manecilla apunta la hora")
            print(f"Major Errors in the placement of the hands (Significantly out of course incluiding 10 to 11){score_m}") 
        
        if flag == 0:
            score_m = 0

    #Si existe diferencia en las manecillas y existe el 2 pero no el 11
    if diferencia_manecillas_bool == True and (2 in numeros_final and 11 not in numeros_final):
        
        #Get Values
        well_placed_2pm = lower_right_corner_2pm<angulo_manecillas_2pm<upper_left_corner_2pm
        well_placed_11pm = 110<angulo_manecillas_11am<130
       
        
        #Si apunta a las 2 y la otra manecilla esta dentro del rango
        if well_placed_2pm == True and well_placed_11pm == True:
            score_m = 4
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas y ambas manecillas apunta a la hora")
            print(f"Hands are in correct position and the size diference is respetected {score_m}")
        
        
        #Se respeta la diferenicia de manecillas pero solo se apunta a hacia a una hora
        if (well_placed_2pm == True and well_placed_11pm == False) or (well_placed_2pm == False and well_placed_11pm == True):
            score_m = 3
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas pero solo apunta hacia una hora")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
        
        #Se respeta la diferencia de longitud de manecillas pero no apunta hacia ninguna hora
        if well_placed_2pm == False and well_placed_11pm == False:
            
            score_m = 2
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas y no apunta hacia ninguna hora")
            print(f"Major Errors in the placement of the hands (Significantly out of course incluiding 10 to 11) {score_m}")  
    
    if diferencia_manecillas_bool == False and (2 in numeros_final and 11 not in numeros_final): 
            #No se respeta la diferencia de medidas y ambas manecillas apuntan a la hora
          #Get Values
        well_placed_2pm = lower_right_corner_2pm<angulo_manecillas_2pm<upper_left_corner_2pm
        well_placed_11pm = 110<angulo_manecillas_11am<130
        
        if well_placed_2pm == True and well_placed_11pm == True:
            score_m = 3
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeto la diferencia de medidas de las manecillas pero apunta ambas horas")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
            
        #No se respeta la diferenicia de manecillas y solo se apunta a hacia a una hora
        if (well_placed_2pm == True and well_placed_11pm == False) or (well_placed_2pm == False and well_placed_11pm == True):
            score_m = 2
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeto la diferencia de las medidas de las manecillas y solo apunta hacia una hora")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
            
        #Se respeta la diferencia de longitud de manecillas pero no apunta hacia ninguna hora
        if well_placed_2pm == False and well_placed_11pm == False:
            score_m= 1
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeta la diferencia de las medidas de las manecillas y ninguna manecilla apunta la hora")
            print(f"Major Errors in the placement of the hands (Significantly out of course incluiding 10 to 11){score_m}") 

        if flag == 0:
            score_m = 0

    #Si existe diferencia en las manecillas
    if diferencia_manecillas_bool == True and (11 in numeros_final and 2 not in numeros_final):
        
    #Analizamos si apunta o no hacia la hora 2
     #Get Values
        well_placed_11pm = upper_right_corner_11am<angulo_manecillas_11am<lower_left_corner_11am
        well_placed_2pm = 20<angulo_manecillas_2pm<40
    
    #Se respeta la diferencia de longitud de manecillas y ambas apuntan a la hora que corresponde
        
        if well_placed_11pm == True and well_placed_2pm == True:
            score_m = 4
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas y ambas manecillas apunta a la hora")
            print(f"Hands are in correct position and the size diference is respetected {score_m}")
        
        
        #Se respeta la diferenicia de manecillas pero solo se apunta a hacia a una hora
        if (well_placed_11pm == True and well_placed_2pm == False) or (well_placed_11pm == False and well_placed_2pm == True):
            score_m = 3
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas pero solo apunta hacia una hora")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
        
        
        #Se respeta la diferencia de longitud de manecillas pero no apunta hacia ninguna hora
        if well_placed_11pm == False and well_placed_2pm == False:
            score_m = 2
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas y no apunta hacia ninguna hora")
            print(f"Major Errors in the placement of the hands (Significantly out of course incluiding 10 to 11) {score_m}") 
                
        #Si no se respeta la diferencia de las medidas de las manecillas 
    if diferencia_manecillas_bool == False and 11 in numeros_final and 2 not in numeros_final:
        #No se respeta la diferencia de medidas y ambas manecillas apuntan a la hora
         #Get Values
        well_placed_11pm = upper_right_corner_11am<angulo_manecillas_11am<lower_left_corner_11am
        well_placed_2pm = 20<angulo_manecillas_2pm<40
        
        if well_placed_11pm == True and well_placed_2pm == True:
            score_m = 3
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeto la diferencia de medidas de las manecillas pero apunta ambas horas")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
                
                
        #No se respeta la diferenicia de manecillas y solo se apunta a hacia a una hora
        if (well_placed_11pm == True and well_placed_2pm == False) or (well_placed_11pm == False and well_placed_2pm == True):
            score_m = 2
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeto la diferencia de las medidas de las manecillas y solo apunta hacia una hora")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
            
            
        #Se respeta la diferencia de longitud de manecillas pero no apunta hacia ninguna hora
        if well_placed_11pm == False and well_placed_2pm == False:
            score_m = 1
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeta la diferencia de las medidas de las manecillas y ninguna manecilla apunta la hora")
            print(f"Major Errors in the placement of the hands (Significantly out of course incluiding 10 to 11) {score_m}")
        
        if flag == 0:
            score_m = 0

    #si no detectó ningun número, evaluamos en funcion de sus angulos y si estan dentro del rango
    if diferencia_manecillas_bool == True and (11 not in numeros_final and 2 not in numeros_final):
        print("Se respeto la distancia y no encontro el numero 11 ni 2")
        well_placed_11pm = 110<angulo_manecillas_11am<130
        well_placed_2pm = 20<angulo_manecillas_2pm<40
        well_placed_angles = 80<angulo_entre_manecillas<100
        
        #Si los angulos estan dentro del rango y entre las manecillas tambien
        if well_placed_11pm == True and well_placed_2pm == True and well_placed_angles == True:
            score_m = 4
        
        if well_placed_11pm == True and well_placed_2pm == False and well_placed_angles == True:
            score_m = 3
            
        if well_placed_11pm == False and well_placed_2pm == True and well_placed_angles == True:
            score_m = 3    
        
        if well_placed_11pm == False and well_placed_2pm == False and well_placed_angles == True:
            score_m = 3    
        
        if well_placed_11pm == True and well_placed_2pm == True and well_placed_angles == False:
            score_m = 3
        
        if well_placed_11pm == True and well_placed_2pm == False and well_placed_angles == False:
            score_m = 2
            
        if well_placed_11pm == False and well_placed_2pm == True and well_placed_angles == False:
            score_m = 2    
        
        if well_placed_11pm == False and well_placed_2pm == False and well_placed_angles == False:
            score_m = 1    
        
    #si no detectó ningun número, evaluamos en funcion de sus angulos y si estan dentro del rango
    if diferencia_manecillas_bool == False and (11 not in numeros_final and 2 not in numeros_final):
        
        well_placed_11pm = 110<angulo_manecillas_11am<130
        well_placed_2pm = 20<angulo_manecillas_2pm<40
        
        #Si los angulos estan dentro del rango y entre las manecillas tambien
        if well_placed_11pm == True and well_placed_2pm == True and 80<angulo_entre_manecillas<100 == True:
            score_m = 4
        
        if well_placed_11pm == True and well_placed_2pm == False and 80<angulo_entre_manecillas<100 == True:
            score_m = 3
            
        if well_placed_11pm == False and well_placed_2pm == True and 80<angulo_entre_manecillas<100 == True:
            score_m = 3    
        
        if well_placed_11pm == False and well_placed_2pm == False and 80<angulo_entre_manecillas<100 == True:
            score_m = 3    

        if well_placed_11pm == True and well_placed_2pm == True and 80<angulo_entre_manecillas<100 == False:
            score_m = 3
        
        if well_placed_11pm == True and well_placed_2pm == False and 80<angulo_entre_manecillas<100 == False:
            score_m = 2
            
        if well_placed_11pm == False and well_placed_2pm == True and 80<angulo_entre_manecillas<100 == False:
            score_m = 2    
        
        if well_placed_11pm == False and well_placed_2pm == False and 80<angulo_entre_manecillas<100 == False:
            score_m = 1 

        if flag == 0:
            score_m = 0


# ----------------- Evaluacion de numeros segun Roleau --------------------- #

#Evaluamos en funcion de lo obtenido
score_numeros = 0
n_correctos_cantidad = len(numeros_correctos)
n_error_espacial = len(numeros_con_error_espacial)
n_detecto_algo = len(numeros_algo_detectado)
n_contorno = len(numeros_desde_contorno)
numeros_afuera_del_reloj = numeros['overlap_percentage']

#Condicionales para números
if (len(numeros_final)>= 8 and numeros_afuera_del_reloj > .8) and (n_detecto_algo <3 or n_error_espacial<3):
    
    score_numeros = 4
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los numeros con errores mínimos")
    print(f"All present in right order and at most minimal error in the patial arrangement: {score_numeros}") 

if (len(numeros_final)>=8 and (numeros_afuera_del_reloj > .8)) and (n_detecto_algo >=4  or n_error_espacial>=3):

    score_numeros = 3
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los números pero no hay distorsiones pequeñas")
    print(f"All Present but errors in spatial arrangement: {score_numeros}") 

if (len(numeros_final)>=8 and (numeros_afuera_del_reloj > .8)) and (n_detecto_algo >=7  or n_error_espacial>=6):
    score_numeros = 2
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Hay numeros faltantes y errores en cuanto a distorsion")
    print(f"Numbers Missing or added but no gross distortions of the remanining numbers: {score_numeros}") 


if (len(numeros_final)< 8 and (numeros_afuera_del_reloj > .8)) and (n_detecto_algo <3 or n_error_espacial<3): 
    score_numeros = 3
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los números pero no hay distorsiones pequeñas")
    print(f"All Present but errors in spatial arrangement: {score_numeros}") 

if (len(numeros_final)<8 and (numeros_afuera_del_reloj > .8)) and (n_detecto_algo >=4  or n_error_espacial>=3):
    score_numeros = 2
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Hay numeros faltantes y errores en cuanto a distorsion")
    print(f"Numbers Missing or added but no gross distortions of the remanining numbers: {score_numeros}") 

if (len(numeros_final)<8 and (numeros_afuera_del_reloj > .8)) and (n_detecto_algo >=7  or n_error_espacial>=6):
    score_numeros = 1
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Hay numeros faltantes y errores en cuanto a distorsion")
    print(f"Missing numbers and gross distorsion errors: {score_numeros}") 

if (len(numeros_final)<= 6 and (numeros_afuera_del_reloj > .8)) and (n_detecto_algo <3 or n_error_espacial<3): 
    score_numeros = 2
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los números pero no hay distorsiones pequeñas")
    print(f"All Present but errors in spatial arrangement: {score_numeros}") 

if (len(numeros_final)<= 6 and (numeros_afuera_del_reloj > .8)) and (n_detecto_algo >=4  or n_error_espacial>=3):
 
    score_numeros = 1
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los números pero no hay distorsiones pequeñas")
    print(f"All Present but errors in spatial arrangement: {score_numeros}") 

if (len(numeros_final)<= 3 and (numeros_afuera_del_reloj > .8)) and (n_detecto_algo <=3  or n_error_espacial<=3): 
    score_numeros = 1
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los números pero no hay distorsiones pequeñas")
    print(f"All Present but errors in spatial arrangement: {score_numeros}") 

if len(numeros_final) <=1:
    score_numeros = 0


#Condicionales para números AFUERA DEL RELOJ
if (len(numeros_final)>= 8 and numeros_afuera_del_reloj <0.5) and (n_detecto_algo <3 or n_error_espacial<3):
    
    score_numeros = 4
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los numeros con errores mínimos")
    print(f"All present in right order and at most minimal error in the patial arrangement: {score_numeros}") 

if (len(numeros_final)>=8 and (numeros_afuera_del_reloj <0.5)) and (n_detecto_algo >=4  or n_error_espacial>=3):

    score_numeros = 3
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los números pero no hay distorsiones pequeñas")
    print(f"All Present but errors in spatial arrangement: {score_numeros}") 

if (len(numeros_final)>=8 and (numeros_afuera_del_reloj <0.5)) and (n_detecto_algo >=7  or n_error_espacial>=6):
        score_numeros = 2
        print("# ------------------------------------- SCORE ------------------------------------- # \n")
        print("Hay numeros faltantes y errores en cuanto a distorsion")
        print(f"Numbers Missing or added but no gross distortions of the remanining numbers: {score_numeros}") 


if (len(numeros_final)< 8 and (numeros_afuera_del_reloj <0.5)) and (n_detecto_algo <3 or n_error_espacial<3): 
    score_numeros =3
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los números pero no hay distorsiones pequeñas")
    print(f"All Present but errors in spatial arrangement: {score_numeros}") 

if (len(numeros_final)<8 and (numeros_afuera_del_reloj <0.5)) and (n_detecto_algo >=4  or n_error_espacial>=3):

    score_numeros = 2
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Hay numeros faltantes y errores en cuanto a distorsion")
    print(f"Numbers Missing or added but no gross distortions of the remanining numbers: {score_numeros}") 

if (len(numeros_final)<8 and (numeros_afuera_del_reloj <0.5)) and (n_detecto_algo >=7  or n_error_espacial>=6):
    score_numeros = 1
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Hay numeros faltantes y errores en cuanto a distorsion")
    print(f"Missing numbers and gross distorsion errors: {score_numeros}") 


if (len(numeros_final)<= 6 and (numeros_afuera_del_reloj <.5)) and (n_detecto_algo <3 or n_error_espacial<3): 
    score_numeros = 2
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los números pero no hay distorsiones pequeñas")
    print(f"All Present but errors in spatial arrangement: {score_numeros}") 

if (len(numeros_final)<= 6 and (numeros_afuera_del_reloj <.5)) and (n_detecto_algo >=4  or n_error_espacial>=3):
 
    score_numeros = 1
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los números pero no hay distorsiones pequeñas")
    print(f"All Present but errors in spatial arrangement: {score_numeros}") 

if (len(numeros_final)<= 3 and (numeros_afuera_del_reloj <.5)) and (n_detecto_algo <=3  or n_error_espacial<=3): 
    score_numeros = 1
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("Se respeta el orden de los números pero no hay distorsiones pequeñas")
    print(f"All Present but errors in spatial arrangement: {score_numeros}") 

if len(numeros_final) <=1:
    print("# ------------------------------------- SCORE ------------------------------------- # \n")
    print("No numbers found on image")
    score_numeros = 0

else:
    score = 0

#Evaluamos Contorno del reloj
image_path,circle_info = process_image_for_contours(image_path='contour/imagen.png')

score_contour = circle_info[0]['score']
print("# ------------------------------------- SCORE ------------------------------------- # \n")
print(f"Puntuacion del contorno: {score_contour}")

puntuacion_final = score_contour + score_m + score_numeros

print("# ------------------------------------- FINAL SCORE ------------------------------------- # \n")
print(str(puntuacion_final) + " Puntos en la escala de Rouleau")


display('nums_detected',imagen_para_numeros,imagen_para_manecillas,imagen_de_contorno,score_m,score_numeros,score_contour,puntuacion_final)
cv2.imshow("Angulos",imagen_para_angulos)
cv2.waitKey(0)
cv2.destroyAllWindows()
vaciar_carpeta('nums_detected')

