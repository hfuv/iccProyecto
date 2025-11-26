import numpy as np
from sklearn import datasets
datos=datasets.load_digits()
w=datos["data"] # con esto se compara
t=datos["target"]
# uso de target y data hay 1797 numeros por comparar cada uno ya etiquetado
print(w.shape)
x=[]# distancias
y=[]# etiquetas
imagen_ingreso=0 # es solo para poner valor falta todavia
for a in range(1797):
    x.append(np.sqrt(np.sum(np.power(imagen_ingreso-w[a],2))))
    y.append(t[a])
s=t[x.index(min(x))] # etiqueta supuesta