import numpy as np
import tensorflow as tf
import cv2
import os

trainPath = "C:/Users/soval/OneDrive/Escritorio/Expresiones/Data/train"
testPath = "C:/Users/soval/OneDrive/Escritorio/Expresiones/Data/test"

folderList = os.listdir(trainPath)
folderList.sort()

print(folderList)

x_train = []
y_train = []

x_test = []
y_test = []

for i , category in enumerate(folderList):
    files = os.listdir(os.path.join(trainPath, category))
    for file  in files:
        print(category+"/"+file)
        img = cv2.imread(os.path.join(trainPath, category, file), 0)
        if img is None:
            print(f"Error al leer imagen: {file}")
            continue
        x_train.append(img)
        y_train.append(i)

print(len(x_train)) #28709
print(y_train)
print(len(y_train))

for i , category in enumerate(folderList):
    files = os.listdir(os.path.join(testPath, category))
    for file  in files:
        print(category+"/"+file)
        img = cv2.imread(os.path.join(testPath, category, file), 0)
        if img is None:
            print(f"Error al leer imagen: {file}")
            continue
        x_test.append(img)
        y_test.append(i)

print("tests")
print(len(x_test))
print(len(y_test))

x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

print(x_train.shape)
print(x_train[0])

x_train = x_train / 255.0
x_test = x_test / 255.0

#reshape

numOfImages = x_train.shape[0]
x_train = x_train.reshape(numOfImages,48,48,1)

print(x_train[0])
print(x_train.shape)

numOfImages = x_test.shape[0]
x_test = x_test.reshape(numOfImages,48,48,1)
print(x_test.shape)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

print("to categorical: ")
print(y_train)
print(y_train.shape)
print(y_train[0])
 
 #build model

input_shape = x_train.shape[1:]
print(input_shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Suponiendo que tenés algo como esto definido
input_shape = (48, 48, 1)  # Cambiar según tus imágenes

# Crear modelo CNN
model = Sequential()

model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_shape))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Bloque 2
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Bloque 3
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Bloque 4
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Bloque 5
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Clasificación
model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dense(7, activation="softmax"))  # 7 clases de emociones

# Mostrar resumen del modelo
model.summary()

# Compilar el modelo
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Hiperparámetros
batch = 32
epochs = 7

# Pasos por época (por si usás generators o ImageDataGenerators)
stepsPerEpoch = int(np.ceil(len(x_train) / batch))
validationSteps = int(np.ceil(len(x_test) / batch))

# Early stopping para evitar overfitting
stopEarly = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Entrenamiento
history = model.fit(
    x_train,
    y_train,
    batch_size=batch,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
    shuffle=True,
    callbacks=[stopEarly]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))  # o range(epochs) si querés fijo



# ----------- Accuracy ----------
plt.plot(epochs_range, acc, 'r', label='Train Accuracy')
plt.plot(epochs_range, val_acc, 'b', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(epochs_range, loss, 'r', label='Train Loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')
plt.show()

modelFileName = "emotion.h5"  # Se guarda en la misma carpeta que Step01
model.save(modelFileName)
print(f"Modelo guardado correctamente en: {os.path.abspath(modelFileName)}")

