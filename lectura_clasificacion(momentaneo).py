import numpy as np
from sklearn import datasets
import cv2
import glob
import pandas as pd
def normalizador(ruta_de_imagen):
   img_array = cv2.imread(ruta_de_imagen, cv2.IMREAD_GRAYSCALE) #"Datasets/miNumero_5_.png" debe ser de esta forma
   img_array=255-img_array
   nueva_img = cv2.resize(img_array, (8, 8))
   nueva_img=np.round((nueva_img/255.0)*16)
   aplanada=nueva_img.flatten().reshape(1, -1) # esta linea es para aplanar me falta entenderla bien
   return aplanada

def clasificador(aplanada,l:int): # actualizando clasificador para 3
   datos=datasets.load_digits()
   w=datos["data"] # con esto se compara
   t=datos["target"]
   x=[]# distancias
   y=[]# etiquetas
   r=[]# lista de cercanos
   for a in range(1797): # por shape
      x.append(np.sqrt(np.sum(np.power(aplanada-w[a],2))))
      y.append(t[a])
   t=list(zip(x, y)) # lista de distancias y su clave
   for q in range(0,l):
       r.append(sorted(t)[q][1])
   return r
# usare archivos en forma serial para poder hacer cambios en la ruta
def preguntas(cantidad_de_cercanos:int,permitir_info:bool):
    d=glob.glob("Datasets/*.png")# la clave verdadera por favor poner con _numero_
    matriz = np.zeros((10, 10))
    p=1
    for a in d:
        t=clasificador(normalizador(a),cantidad_de_cercanos)
        if permitir_info:
           print("las etiquetas de las distancias del dato "+str(p)+" es "+str(t),"su verdadera etiqueta es:"+str(a.split("_")[-2]))
        diccionario_auxiliar={}
        for i in t:
            diccionario_auxiliar[i]=0
        for i in t:
            diccionario_auxiliar[i]+=1
        n=list(diccionario_auxiliar.values())
        valores=list(diccionario_auxiliar.items())
        for llave,valor in valores:
           if valor>=len(t)/2 and permitir_info :
               print("Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número:", str(llave))
               matriz[int(a.split("_")[-2])][llave] +=1
               break
           elif valor==max(n) and permitir_info: # se ve la cantidad de veces aparece una clave
               print("Soy la inteligencia artificial, y no hay un dato dominante por lo tanto tomare el valor mayor:",
                     str(llave))
               matriz[int(a.split("_")[-2])][llave] += 1
               break
           if not permitir_info:
               if valor >= len(t) / 2:
                   matriz[int(a.split("_")[-2])][llave] += 1
                   break
               elif valor == max(n) :
                   matriz[int(a.split("_")[-2])][llave] += 1
                   break
        p+=1
    etiquetas_guia=["0_supu","1_supu","2_supu","3_supu","4_supu","5_supu","6_supu","7_supu","8_supu","9_supu"]
    etiquetas_guia02= ["0_real", "1_real", "2_real", "3_real", "4_real", "5_real", "6_real", "7_real", "8_real", "9_real"]
    matriz_lectura=pd.DataFrame(matriz,index=etiquetas_guia02,columns=etiquetas_guia)
    pd.set_option('display.max_rows', None) # para ver asi sin mas
    pd.set_option('display.max_columns', None) # para ver asi sin mas
    return matriz_lectura,matriz

def matriz_2(numero:int,cercanos:int,info:bool): # analisis de un numero que se quiera
    r,s=preguntas(cercanos,info)
    matriz=np.zeros((2,2))
    matriz[0,0]=s[numero][numero] # verdadero positivo
    matriz[1,0]=np.sum(s[:,numero])-matriz[0,0]# falso positivo
    matriz[0,1]=np.sum(s[numero,:])-matriz[0,0] # falso negativo
    matriz[1,1]=np.sum(s)-matriz[0,0]-matriz[0,1]-matriz[1,0]# verdadero negativo
    etiqueta_fila=["real_x","real_y"]
    etiqueta_columna = ["predict_x", "predict_y"]
    matriz_confusion = pd.DataFrame(matriz, index=etiqueta_fila, columns=etiqueta_columna)
    accuracy=(matriz[0,0]+matriz[1,1])/np.sum(matriz)
    precision=matriz[0,0]/(matriz[0,0]+matriz[1,0])
    recall=matriz[0,0]/(matriz[0,0]+matriz[0,1])
    F1_Score=2*(precision*recall)/(precision+recall)
    return matriz_confusion,{'accuracy':accuracy,'precision':precision,'recall':recall,'F1 Score':F1_Score}
print(preguntas(3,False))