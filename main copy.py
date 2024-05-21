# ------------------------ Importar Librerias ----------------------- #
import runpy
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import shutil
import math

#Graficamos todos lo números encontrados que se guerdan en la carpeta
def display(folder,img1,img2,img3,score_manecillas):
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
    axs2[0].axis('off')  # Desactiva los ejes para una visualización más limpia
       
    axs2[1].imshow(img2_rgb, cmap='gray')
    axs2[1].set_title('Manecillas Detectadas')
    axs2[1].set_xlabel('Puntuacion ' + str(score_manecillas))  # Título para el eje x del tercer subplot

    axs2[2].imshow(img3_rgb, cmap='gray')
    axs2[2].set_title('Contorno')
    axs2[2].axis('off')

    # Ajusta el layout para evitar que los títulos se solapen
    fig2.tight_layout()
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
manecillas = runpy.run_path('contour_segmentation3.py')
numeros = runpy.run_path('results_yolo.py')

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

#Si hay estan en distinto lado
if angulo_entre_manecillas>60:
    longitud_manecilla_2pm = manecillas['line_lenghts'][0]
    longitud_manecilla_11am = manecillas['line_lenghts'][1]
    diferencia_manecillas_bool = longitud_manecilla_11am < longitud_manecilla_2pm
    diferencia_manecillas = longitud_manecilla_2pm-longitud_manecilla_11am
else:
    #Longitud de manecilla y angulo se recorre porque la manecilla esta del lado izquierda
    longitud_manecilla_11am = manecillas['line_lenghts'][0]
    longitud_manecilla_2pm = manecillas['line_lenghts'][1]
    angulo_manecillas_11am = manecillas["clockhand_angle_2"]
    diferencia_manecillas_bool = longitud_manecilla_11am < longitud_manecilla_2pm
    diferencia_manecillas = longitud_manecilla_2pm-longitud_manecilla_11am
    
# -------------------------- Números -------------------------- #
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
            
            coordenadas_2pm= (x,x+w,y,y+h)

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
            
# Suponemos que numeros['class_names'] es una lista que contiene algunos números detectados.
numeros_detectados = numeros['class_names']
print("Números detectados:", numeros_detectados)

# Lista original de números a detectar
numeros_a_detectar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Números no detectados
numeros_no_detectados = [n for n in numeros_a_detectar if n not in numeros_detectados]
print("Números no detectados:", numeros_no_detectados)




#Numeros detectados correctamente
numeros_detectados_correcto = numeros['detected_correct']

#Números detectados con error espacial
numeros_detectados_con_error_espacial= numeros['detected_in_wrong_arrangement']

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

print(f"Etiquetas{etiquetas}")
score_m  = 0

#Si no hay lineas la puntuación es 0
if not lineas_manecillas:
    print("No hay Manecillas detectadas")
    score_m = 0    

#Si se detectaron dos manecillas y hay elementos
elif len(lineas_manecillas) == 2 or len(lineas_manecillas) == 3:
    print(f"Longitud Manecilla 2 {longitud_manecilla_2pm}")
    print(f"Longitud Manecilla 11 {longitud_manecilla_11am}")
    print(f"Angulo entre manecillas {angulo_entre_manecillas}")
    
    well_placed_2pm = lower_right_corner_2pm<angulo_manecillas_2pm<upper_left_corner_2pm
    well_placed_11pm = upper_right_corner_11am<angulo_manecillas_11am<lower_left_corner_11am
    
    print(lower_right_corner_2pm,angulo_manecillas_2pm,upper_left_corner_2pm)
    print(upper_right_corner_11am,angulo_manecillas_11am,lower_left_corner_11am)
    
    #Si existe diferencia en las manecillas y apunta a ambas horas
    if diferencia_manecillas_bool == True and 2 in numeros_detectados and 11 in numeros_detectados:
        
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
    else:
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



    #Si existe diferencia en las manecillas
    if diferencia_manecillas_bool == True and 11 not in numeros_detectados:
        
        well_placed_2pm = lower_right_corner_2pm<angulo_manecillas_2pm<upper_left_corner_2pm
        #Si apunta a las 2 y la otra manecilla esta dentro del rango
        if well_placed_2pm == True and 110<angulo_manecillas_11am<130 == True:
            score_m = 4
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas y ambas manecillas apunta a la hora")
            print(f"Hands are in correct position and the size diference is respetected {score_m}")
        
        
            #Se respeta la diferenicia de manecillas pero solo se apunta a hacia a una hora
        if (well_placed_2pm == True and 110<angulo_manecillas_11am<130 == False) or (well_placed_2pm == False and 110<angulo_manecillas_11am<130 == True):
            score_m = 3
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas pero solo apunta hacia una hora")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
        
        #Se respeta la diferencia de longitud de manecillas pero no apunta hacia ninguna hora
        if well_placed_2pm == False and 110<angulo_manecillas_11am<130 == False:
            score_m = 2
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas y no apunta hacia ninguna hora")
            print(f"Major Errors in the placement of the hands (Significantly out of course incluiding 10 to 11) {score_m}")  
    
    else:
            #No se respeta la diferencia de medidas y ambas manecillas apuntan a la hora
        if well_placed_2pm == True and 110<angulo_manecillas_11am<130 == True:
            score_m = 3
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeto la diferencia de medidas de las manecillas pero apunta ambas horas")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
            
        #No se respeta la diferenicia de manecillas y solo se apunta a hacia a una hora
        if (well_placed_2pm == True and 110<angulo_manecillas_11am<130 == False) or (well_placed_2pm == False and 110<angulo_manecillas_11am<130 == True):
            score_m = 2
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeto la diferencia de las medidas de las manecillas y solo apunta hacia una hora")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
            
        #Se respeta la diferencia de longitud de manecillas pero no apunta hacia ninguna hora
        if well_placed_2pm == False and 110<angulo_manecillas_11am<130 == False:
            score_m= 1
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeta la diferencia de las medidas de las manecillas y ninguna manecilla apunta la hora")
            print(f"Major Errors in the placement of the hands (Significantly out of course incluiding 10 to 11){score_m}") 


        
    #Si existe diferencia en las manecillas
    if diferencia_manecillas_bool == True and 2 not in numeros_detectados:
        
        well_placed_11pm = upper_right_corner_11am<angulo_manecillas_11am<lower_left_corner_11am
        
        if well_placed_11pm == True and 80<angulo_manecillas_2pm<100 == True:
            score_m = 4
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas y ambas manecillas apunta a la hora")
            print(f"Hands are in correct position and the size diference is respetected {score_m}")
        
        
        #Se respeta la diferenicia de manecillas pero solo se apunta a hacia a una hora
        if (well_placed_11pm == True and 80<angulo_manecillas_2pm<100 == False) or (well_placed_11pm == False and 80<angulo_manecillas_2pm<100 == True):
            score_m = 3
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas pero solo apunta hacia una hora")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
        
        
        #Se respeta la diferencia de longitud de manecillas pero no apunta hacia ninguna hora
        if well_placed_11pm == False and 80<angulo_manecillas_2pm<100 == False:
            score_m = 2
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("Se respeto la diferencia de las medidas de las manecillas y no apunta hacia ninguna hora")
            print(f"Major Errors in the placement of the hands (Significantly out of course incluiding 10 to 11) {score_m}") 
                
        #Si no se respeta la diferencia de las medidas de las manecillas 
    elif diferencia_manecillas_bool == False:
        #No se respeta la diferencia de medidas y ambas manecillas apuntan a la hora
        if well_placed_11pm == True and 80<angulo_manecillas_2pm<100 == True:
            score_m = 3
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeto la diferencia de medidas de las manecillas pero apunta ambas horas")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
                
                
        #No se respeta la diferenicia de manecillas y solo se apunta a hacia a una hora
        if (well_placed_11pm == True and 80<angulo_manecillas_2pm<100 == False) or (well_placed_11pm == False and 80<angulo_manecillas_2pm<100 == True):
            score_m = 2
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeto la diferencia de las medidas de las manecillas y solo apunta hacia una hora")
            print(f"Slight error in the placement of the hands or no representation of size difference between the hands {score_m}")
            
            
        #Se respeta la diferencia de longitud de manecillas pero no apunta hacia ninguna hora
        if well_placed_11pm == False and 80<angulo_manecillas_2pm<100 == False:
            score_m = 1
            print("# ------------------------------------- SCORE ------------------------------------- # \n")
            print("No se respeta la diferencia de las medidas de las manecillas y ninguna manecilla apunta la hora")
            print(f"Major Errors in the placement of the hands (Significantly out of course incluiding 10 to 11){score_m}") 

    #si no detectó ningun número, evaluamos en funcion de sus angulos y si estan dentro del rango
    else:
        well_placed_11pm = 110<angulo_manecillas_11am<130
        well_placed_2pm = 80<angulo_manecillas_2pm<100
        
        #Si los angulos estan dentro del rango y entre las manecillas tambien
        if well_placed_11pm == True and well_placed_2pm == True and 80<angulo_entre_manecillas<100 == True:
            score_m = 4
        #Si los angulos de las manecillas estan dentro del rango ideal 
        if well_placed_11pm == True and well_placed_2pm == True and 80<angulo_entre_manecillas<100 == False:
            score_m = 3
        if well_placed_11pm == False and well_placed_2pm == False and 80<angulo_entre_manecillas<100 == True:
            score_m = 3
        if well_placed_11pm == False and well_placed_2pm == False and 80<angulo_entre_manecillas<100 == False:
            score_m = 2
        
if len(lineas_manecillas) == 0:
    score_m = 0
#Graficamos todo los resultados

display('nums_detected',imagen_para_numeros,imagen_para_manecillas,imagen_de_contorno,score_m)
cv2.imshow("Angulos",imagen_para_angulos)
cv2.waitKey(0)
cv2.destroyAllWindows()
vaciar_carpeta('nums_detected')

