import numpy as np
from sklearn import datasets
import cv2
img_array = cv2.imread("Datasets/miNumero.png", cv2.IMREAD_GRAYSCALE)
img_array=255-img_array
nueva_img = cv2.resize(img_array, (8, 8))
nueva_img=np.round((nueva_img/255.0)*16)
aplanada=nueva_img.flatten().reshape(1, -1) # esta linea es para aplanar
#-----------------------------------------------------------------------------------------
datos=datasets.load_digits()
w=datos["data"] # con esto se compara
t=datos["target"]
# uso de target y data hay 1797 numeros por comparar cada uno ya etiquetado
print(w.shape)
x=[]# distancias
y=[]# etiquetas
#imagen_ingreso=0 # es solo para poner valor falta todavia
for a in range(1797):
    x.append(np.sqrt(np.sum(np.power(aplanada-w[a],2))))
    y.append(t[a])
s=t[x.index(min(x))] # etiqueta supuesta
print(s)
