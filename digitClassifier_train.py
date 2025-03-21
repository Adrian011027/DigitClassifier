
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Cargar y preparacion de datos
os.chdir(r"C:\Users\angel\Desktop\CUCEI\INCO 9\SIA2\P1")
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

X = train_data.drop(columns='label')
y = train_data['label']

# slpit en conjunto de entrenamiento, validación y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

#normalizacion y preprocesamiento de datos
def prep_dataset(X, y, shape):
    X_prep = np.array(X).reshape((len(X), shape))
    X_prep = X_prep.astype('float32') / 255  
    y_prep = to_categorical(np.array(y))
    return (X_prep, y_prep)

X_train_prep, y_train_prep = prep_dataset(X_train, y_train, 784)
X_val_prep, y_val_prep = prep_dataset(X_val, y_val, 784)
X_test_prep, y_test_prep = prep_dataset(X_test, y_test, 784)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

print("\nEntrenando con Adam...\n")

network = models.Sequential()
network.add(layers.Dense(200, activation='relu', input_shape=(784,)))
network.add(layers.Dense(150, activation='relu'))
network.add(layers.Dense(100, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

# ✅ Compilar el modelo
network.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)


history = network.fit(
    X_train_prep, y_train_prep,
    epochs=12,
    validation_data=(X_val_prep, y_val_prep),
    verbose=1
)

# Evaluacion y guuardar modelo
test_loss, test_acc = network.evaluate(X_test_prep, y_test_prep, verbose=1)
print(f"\nPrecisión en test con Adam: {test_acc:.4f}")

network.save("modelo_numeros_adam.h5")
print("\nModelo guardado como 'modelo_numeros_adam.h5'")

#Graficas y metricas
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

plt.show()

print("\nGenerando la matriz de confusión...")

y_pred = network.predict(X_test_prep)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test_prep, axis=1)


conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')
plt.title("Matriz de Confusion")
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.show()

# Reporte de metricas
print("\nReporte de metricas:")
print(classification_report(y_true_labels, y_pred_labels))
