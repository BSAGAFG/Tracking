import cv2
import subprocess

print("detect opencv: ",cv2.__version__)
# Especifica la ruta al script detect.py y al modelo YOLO que deseas utilizar
detect_script_path = "./detect.py"
model_path = "./yolov7-e6e.pt"

# Especifica la ruta a la imagen video, streaming, que deseas analizar
image_path = "./data/images/Circunvalar5.jpg"

#python detect.py --weights yolov7-e6e.pt --source ./data/images/Circunvalar7.jpgtea

# Ejecuta el script detect.py como un proceso externo
cmd = f"python {detect_script_path} --weights {model_path} --source {image_path} "
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)

# Lee la salida del proceso y haz algo con ella (por ejemplo, imprime los resultados)
output, error = process.communicate()
print(output.decode())

