#usar  yolov7 en python con visual studio code?
import cv2
import sys
sys.path.insert(0, './yolov7')

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, plot_one_box


# Cargar el modelo pre-entrenado
weights_path = 'yolov7/yolov7-e6e.pt'
model = attempt_load(weights_path, map_location=torch.device('gpu'))
names = ['clase1', 'clase2', 'clase3']

# Leer la imagen

# Configura la transformación de la imagen de entrada
img_size = 640
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_path = 'yolov7/data/images/Circunvalar5.jpg'

# imagen_opencv = cv2.imread(image_path)
# imagen_opencv = np.array(imagen_opencv)

# alto, ancho = imagen_opencv.shape[:2]
# nuevo_alto = (alto // 32) * 32
# nuevo_ancho = (ancho // 32) * 32
# imagen = cv2.resize(imagen_opencv, (nuevo_ancho, nuevo_alto))

# Carga la imagen
img = Image.open(image_path).convert('RGB')

# Redimensiona y normaliza la imagen
img, _, _ = letterbox(img, new_shape=img_size)
img = transform(img).unsqueeze(0)

# Detecta objetos en la imagen
detections = model(img)[0]
detections = non_max_suppression(detections, 0.3, 0.5)

# Dibuja los cuadros de detección en la imagen
img = np.array(img.squeeze(0).permute(1,2,0))
for det in detections:
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[:2], det[:, :4], img_size).round()
        for *xyxy, conf, cls in reversed(det):
            plot_one_box(xyxy, img, label=names[int(cls)], color=(255, 0, 0))

# Guarda la imagen con los cuadros de detección
img = Image.fromarray(img)
img.save('output.jpg')

