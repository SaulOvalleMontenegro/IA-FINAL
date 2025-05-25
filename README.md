# IA-FINAL
Este proyecto utiliza una red neuronal convolucional (CNN) entrenada en imágenes faciales para detectar emociones humanas en tiempo real desde una webcam. Los resultados se muestran en una ventana de cámara en vivo y en una interfaz web ligera que refleja la emoción actual detectada.

Detección facial en tiempo real con Haar Cascades.
Clasificación de emociones con modelo CNN (`emotion.h5`) entrenado en el dataset FER2013.
Interfaz web local para visualizar la emoción actual sin necesidad de Flask.
Eficiente y compatible con equipos de gama media.

Debido a limitaciones de espacio, el modelo `.h5` y el dataset completo **no están incluidos en este repositorio**. Pueden descargarlos desde el siguiente enlace:
https://drive.google.com/drive/folders/1STNQphPtOz2URsnKLYLaRhcGA-QCc-Up?usp=sharing

El contenido de dicho drive es el siguiente:
emotion.h5`: modelo entrenado con FER2013.
Data/train`: dataset clasificado por carpetas (`angry`, `happy`, etc.).
