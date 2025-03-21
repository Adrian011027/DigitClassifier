import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from drawNum import dibujar_numero
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tkinter as tk

ruta_imagen = "./imagen_prueba.jpg"

def cargar_y_preprocesar_imagen(ruta):
    # Normalizar, redimensionar y procesar la imagen
    img = Image.open(ruta).convert('L') 
    img = img.resize((28, 28))            
    img_array = np.array(img)    
    img_array = img_array.astype('float32') / 255.0  
    img_array = img_array.reshape(1, 784)  # Redimensionar para el modelo (1, 784 neuronas de entrada)
    
    return img_array, img

def evaluar_imagen():
    modelo = load_model("modelo_numeros_adam.h5")
    dibujar_numero()
    img_array, img_visual = cargar_y_preprocesar_imagen(ruta_imagen)
    prediccion = modelo.predict(img_array)
    etiqueta_predicha = np.argmax(prediccion, axis=1)

    print(f'\nEl número para la imagen es: {etiqueta_predicha[0]}')

    # Crear la figura y mostrar la imagen
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img_visual, cmap='gray')
    plt.title(f"Predicción: {etiqueta_predicha[0]}")
    plt.axis('off')  # Quitar los ejes

    # Centrar la ventana de la imagen
    mng = plt.get_current_fig_manager()
    mng.window.geometry(f"+{int(mng.window.winfo_screenwidth() / 2 - fig.get_figwidth() * 50)}+{int(mng.window.winfo_screenheight() / 2 - fig.get_figheight() * 50)}")

    plt.show()

    # Mostrar porcentajes
    print("\nPorcentajes por clase:")
    porcentajes = prediccion[0] * 100 
    for i, porcentaje in enumerate(porcentajes):
        print(f"Clase {i}: {porcentaje:.2f}%")

if __name__ == "__main__":
    evaluar_imagen()
