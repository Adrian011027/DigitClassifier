from PIL import Image, ImageDraw
import tkinter as tk

def dibujar_numero():
    def guardar_imagen():
        imagen.save("imagen_prueba.jpg")
        print("\nImagen guardada como 'imagen_prueba.jpg'")
        ventana.destroy()

    def limpiar_canvas():
        canvas.delete("all")
        draw.rectangle((0, 0, 280, 280), fill="black")

    def dibujar(event):
        global last_x, last_y
        last_x, last_y = event.x, event.y

    def trazar(event):
        global last_x, last_y
        x, y = event.x, event.y
        canvas.create_line(last_x, last_y, x, y, fill="white", width=15, capstyle=tk.ROUND, smooth=True)
        draw.line([last_x, last_y, x, y], fill="white", width=15)
        last_x, last_y = x, y

   
    ventana = tk.Tk()
    ventana.title("Dibuja un número")
    ventana.geometry("300x350")

    wtotal = ventana.winfo_screenwidth()
    htotal = ventana.winfo_screenheight()
    wventana = 300
    hventana = 350 
    pwidth = round(wtotal/2-wventana/2)
    pheight = round(htotal/2-hventana/2)

    ventana.geometry(str(wventana)+"x"+str(hventana)+"+"+str(pwidth)+"+"+str(pheight))


    canvas = tk.Canvas(ventana, width=280, height=280, bg="black")
    canvas.pack(pady=10)

   
    imagen = Image.new("RGB", (280, 280), "black")
    draw = ImageDraw.Draw(imagen)
    canvas.bind("<Button-1>", dibujar)
    canvas.bind("<B1-Motion>", trazar)

    btn_guardar = tk.Button(ventana, text="Guardar", command=guardar_imagen)
    btn_guardar.pack(side="left", padx=20)

    btn_limpiar = tk.Button(ventana, text="Limpiar", command=limpiar_canvas)
    btn_limpiar.pack(side="right", padx=20)

    ventana.mainloop()