import numpy as np
from sklearn import datasets
import cv2 # ver imagenes
import glob # glob servira para la lectura de rutas
import pandas as pd # ver los resultados
def normalizador(ruta_de_imagen):
   img_array = cv2.imread(ruta_de_imagen, cv2.IMREAD_GRAYSCALE) #"Datasets/(un nombre cualquiera)_numero que es _.png" es de esta forma como debe escribirse
   img_array=255-img_array # se invierten la escala
   nueva_img = cv2.resize(img_array, (8, 8)) # se hace un reescalado
   nueva_img=np.round((nueva_img/255.0)*16) # se normaliza la imagen
   aplanada=nueva_img.reshape(1, -1) # se aplana la imagen
   # reshape(fila,columnas) entonces se pide fila 1 y columna -1 para que alcancen todos los elementos
   return aplanada

def clasificador(aplanada,l:int): # actualizando clasificador para 3
   datos=datasets.load_digits() # se carga los digitos
   w=datos["data"] # con esto se compara uso de shape-> (1797,64)
   t=datos["target"] # en target
   x=[]# distancias
   y=[]# etiquetas
   r=[]# lista de cercanos
   for a in range(1797): # por shape // da las distancia euclideana la formula utilizada
      x.append(np.sqrt(np.sum(np.power(aplanada-(w[a].reshape(1, -1)),2)))) # se .reshape(1, -1) debido al Broadcasting de Numpy que soluciono eso
      # uso de operacions de numpy power para potencias y listo
      y.append(t[a])
   t=list(zip(x, y)) # lista de distancias y su clave se hace zip para despues ordenar
   for q in range(0,l): #cantidad de distancias a usar
       r.append(sorted(t)[q][1]) # a la lista ordenada mediante [1] se accede al segundo elemento
       # como esta ordenado te dara las etiquetas de los mas cercanos
   return r
# usare archivos en forma serial para poder hacer cambios en la ruta
def preguntas(cantidad_de_cercanos:int,permitir_info:bool,permiso_registro:bool):
    d=glob.glob("Datasets/*.png")# la clave verdadera por favor poner con _numero_
    matriz = np.zeros((10, 10))
    p=1 # para que se pueda dar a entender los numeros que entran
    for a in d: # d contiene una lista de la ruta de los archivos
        t=clasificador(normalizador(a),cantidad_de_cercanos) # se aplica clasificador ya que necesitamos las etiquetas ordenadas
        if permitir_info:
           print("las etiquetas de las distancias del dato "+str(p)+" es "+str(t),"su verdadera etiqueta es:"+str(a.split("_")[-2]))
        diccionario_auxiliar={}
        for i in t: # recordar t ya esta ordenado
            diccionario_auxiliar[i]=0 # se inicializa el diccionario
        for i in t:
            diccionario_auxiliar[i]+=1 # se ve quienes se repiten mas
        n=list(diccionario_auxiliar.values())  # guardamos los valores
        valores=list(diccionario_auxiliar.items()) # lista de tuplas(keys,valores) conservando orden
        for llave,valor in valores:
           if valor>=len(t)/2 and permitir_info :
               print("Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número:", str(llave))
               matriz[int(a.split("_")[-2])][llave] +=1
               break
           elif valor==max(n) and permitir_info: # se ve la cantidad de veces aparece una clave
               # con max se busca conseguir el valor por si no hay otro
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
        # cabeceras
    etiquetas_guia=["0_supu","1_supu","2_supu","3_supu","4_supu","5_supu","6_supu","7_supu","8_supu","9_supu"]
    etiquetas_guia02= ["0_real", "1_real", "2_real", "3_real", "4_real", "5_real", "6_real", "7_real", "8_real", "9_real"]
    matriz_lectura=pd.DataFrame(matriz,index=etiquetas_guia02,columns=etiquetas_guia)
    if permiso_registro:
        matriz_lectura.to_csv("Matriz_Confusion1(detalles).csv")
    pd.set_option('display.max_rows', None) # para ver asi sin mas
    pd.set_option('display.max_columns', None) # para ver asi sin mas
    return matriz_lectura,matriz

def matriz_2(numero:int,cercanos:int,info:bool,permiso_registro:bool): # analisis de un numero que se quiera
    r,s=preguntas(cercanos,info,permiso_registro)
    # operaciones entre arrays
    matriz=np.zeros((2,2))
    matriz[0,0]=s[numero][numero] # verdadero positivo
    matriz[1,0]=np.sum(s[:,numero])-matriz[0,0]# falso positivo
    matriz[0,1]=np.sum(s[numero,:])-matriz[0,0] # falso negativo
    matriz[1,1]=np.sum(s)-matriz[0,0]-matriz[0,1]-matriz[1,0]# verdadero negativo
    # cabeceras
    etiqueta_fila=["real_positivo","real_negativo"]
    etiqueta_columna = ["predict_positivo", "predict_negativo"]
    matriz_confusion = pd.DataFrame(matriz, index=etiqueta_fila, columns=etiqueta_columna)
    accuracy=(matriz[0,0]+matriz[1,1])/np.sum(matriz)
    if matriz[0,0]+matriz[1,0]==0:
        print("no hay datos validos en precision")
        precision=0.0
    else:
        precision=matriz[0,0]/(matriz[0,0]+matriz[1,0])
    if matriz[0,0]+matriz[0,1]==0:
        print("no hay datos validos en recall")
        recall=0.0
    else:
        recall=matriz[0,0]/(matriz[0,0]+matriz[0,1])
    if precision==0 or recall==0 or precision+recall==0:
        print("no hay datos validos en F1_Score")
        F1_Score=0.0
    else:
        F1_Score=2*(precision*recall)/(precision+recall)
    if permiso_registro:
        matriz_confusion.to_csv("Matriz_Confusion2(detalles).csv")
    return matriz_confusion,{'accuracy':accuracy,'precision':precision,'recall':recall,'F1 Score':F1_Score}
print(preguntas(3,False,False))