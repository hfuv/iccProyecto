import cv2
import numpy as np
img_array = cv2.imread("Datasets/miNumero.png", cv2.IMREAD_GRAYSCALE)
img_array=255-img_array
nueva_img = cv2.resize(img_array, (8, 8))
nueva_img=np.round((nueva_img/255.0)*16)
aplanada=nueva_img.flatten().reshape(1, -1) # esta linea es para aplanar
