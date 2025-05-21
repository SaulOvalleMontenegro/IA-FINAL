import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
import cv2
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr

# Ruta al modelo entrenado
model_file = "C:/Users/soval/OneDrive/Escritorio/Expresiones/emotion.h5"
model = tf.keras.models.load_model(model_file)

# Ver resumen del modelo cargado
model.summary()

# Obtener las categorías (emociones)
print("Categories:")
trainPath = "C:/Users/soval/OneDrive/Escritorio/Expresiones/Data/train"
categories = os.listdir(trainPath)
categories.sort()
print(categories)
numOfClasses = len(categories)
print("Número de clases:", numOfClasses)

# Cargar Haar Cascade
haarcascadefile = "C:/Users/soval/OneDrive/Escritorio/Expresiones/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascadefile)

# Función para preparar la imagen para el modelo
def prepareImageForModel(grayFace):
    resized = cv2.resize(grayFace, (48, 48))
    normalized = resized.astype("float32") / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped

# Configuración de correo
SENDER_EMAIL = "ecotonernotificaciones@gmail.com"
APP_PASSWORD = "jgsq rlkd uxld vilu"
RECEIVER_EMAIL = "diegocosillo@gmail.com"

def enviar_alerta():
    subject = "Alerta: Nadie esta prestando atencion"
    body = "Es posible que los alumnos no esten poniendo atencion. Se recomienda hacer una actividad interactiva."

    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["From"] = formataddr((str(Header("Detector de Atencion", "utf-8")), SENDER_EMAIL))
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = Header(subject, "utf-8")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("Correo de alerta enviado")

    except Exception as e:
        print("Error al enviar correo:", e)

# Captura de video desde la webcam
cap = cv2.VideoCapture(1)  # 0 para la cámara por defecto

# Control de detección y tiempo
last_face_time = time.time()
alerta_enviada = False
rostro_detectado_anteriormente = True  # Asumimos que hay rostro al inicio

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # Mostrar mensaje si no hay rostros detectados
        cv2.putText(frame, "Los estudiantes no estan poniendo atencion",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if rostro_detectado_anteriormente:
            last_face_time = time.time()
            rostro_detectado_anteriormente = False

        if time.time() - last_face_time >= 10 and not alerta_enviada:
            enviar_alerta()
            alerta_enviada = True
    else:
        for (x, y, w, h) in faces:
            roiGray = gray[y:y+h, x:x+w]
            imgForModel = prepareImageForModel(roiGray)
            prediction = model.predict(imgForModel)
            predictedLabel = categories[np.argmax(prediction)]

            # Guardar emoción en archivo
            with open("estado_emocion.txt", "w") as f:
                f.write(predictedLabel)

            # Dibujar el rectángulo y la emoción predicha
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, predictedLabel, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            break  # Solo usamos la primera cara detectada

        last_face_time = time.time()
        alerta_enviada = False
        rostro_detectado_anteriormente = True

    # Mostrar el frame con la predicción o mensaje
    cv2.imshow("Reconocimiento de emociones", frame)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
