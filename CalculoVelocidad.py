import cv2
import darknet

# Cargar los pesos y la configuración de YOLOv7
net = darknet.load_net("yolov7.cfg", "yolov7.weights", 0)
meta = darknet.load_meta("coco.data")

# Leer el video
cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Inicializar el rastreador
tracker = cv2.TrackerKCF_create()

# Detectar objetos en cada cuadro del video
while True:
    ret, frame = cap.read()
    if not ret:
        break
   
    # Detectar objetos en el cuadro actual
    detections = darknet.detect_image(net, meta, frame)
   
    # Rastrear el objeto de interés
    for detection in detections:
        # Si el objeto detectado es el objeto de interés
        if detection[0] == b'object_of_interest':
            x, y, w, h = detection[2]
            bbox = (int(x), int(y), int(w), int(h))
            ok = tracker.init(frame, bbox)
   
    # Actualizar la posición del objeto rastreado
    ok, bbox = tracker.update(frame)
    if ok:
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
       
        # Calcular la velocidad del objeto
        velocity = (w / fps) * 3.6 # m/s a km/h
       
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()