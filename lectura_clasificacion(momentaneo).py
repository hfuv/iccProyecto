import numpy as np
from sklearn import datasets
import cv2
import glob
def normalizador(ruta_de_imagen):
   img_array = cv2.imread(ruta_de_imagen, cv2.IMREAD_GRAYSCALE) #"Datasets/miNumero.png" debe ser de esta forma
   img_array=255-img_array
   nueva_img = cv2.resize(img_array, (8, 8))
   nueva_img=np.round((nueva_img/255.0)*16)
   aplanada=nueva_img.flatten().reshape(1, -1) # esta linea es para aplanar
   return aplanada

def clasificador(aplanada,l:int): # actualizando clasificador para 3
   datos=datasets.load_digits()
   w=datos["data"] # con esto se compara
   t=datos["target"]
   x=[]# distancias
   y=[]# etiquetas
   r=[]# lista de cercanos
#imagen_ingreso=0 # es solo para poner valor falta todavia el 1797 es de target
   for a in range(1797): # por shape
      x.append(np.sqrt(np.sum(np.power(aplanada-w[a],2))))
      y.append(t[a])
   t=list(zip(x, y))
   for q in range(0,l):
       r.append(sorted(t)[q][1])
   return r



# usare archivos en forma serial para poder hacer cambios en la ruta
# no esta listo
def matriz(cantidad_de_cercanos:int):
    d=glob.glob("Datasets/*.png")# la clave verdadera por favor poner con _numero_
    matriz = np.zeros((10, 10))
    for a in d:
        t=clasificador(normalizador(a),cantidad_de_cercanos)
        # contador=0
        for i in t:
            matriz[5][i]+=1 # cocatenar con panda con las cabeceras
#           if i==a.split("_")[-2]:
#               contador+=1
    return matriz
print(matriz(3))


