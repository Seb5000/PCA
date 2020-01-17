import dlib
import numpy as np
import cv2
import os
import sys

#Crea una ventana de Dlib
window = dlib.image_window()
#nombre de la carpeta con las imagenes de origen
carpeta ="img"
#Nombre de la carpeta de destino donde se guardara las imagenes
# recortadas
nCarpeta = "imgC"

#Inicia el detector de rostros
detector = dlib.get_frontal_face_detector()
#Se carga el modelo para la prediccion
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

for archivo in os.listdir(carpeta):
	# Une el nombre del archivo con el nombre de la carpeta
	dirImg = os.path.join(carpeta, archivo)
	# Carga la imagen dirImg en img
	img = dlib.load_rgb_image(dirImg)
	#Se realiza la deteccion de rostros en img
	dets=detector(img,1)
	#Se guarda informacion sobre el rostro detectado
	cara = sp(img, dets[0])
	#Se recorta la imagen centrandola en el rostro
	imagenr = dlib.get_face_chip(img, cara, size=640)
	#Actualiza la ventana window con imagenr
	window.set_image(imagenr)
	#Se genera la nueva ruta para guardar las imagenes
	nDirImg = os.path.join(nCarpeta, archivo)
	#Se guarda la nueva imagen con OpenCV
	cv2.imwrite(nDirImg, cv2.cvtColor(imagenr, cv2.COLOR_RGB2BGR))

