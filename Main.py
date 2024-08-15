import threading
import time
from djitellopy import Tello
import cv2
from ultralytics import YOLO
import numpy as np
from sort import Sort
import os
import sqlite3
from flask import Flask, render_template, Response

# Inicializar variables
xmin2, ymin2, xmax2, ymax2 = 0, 0, 0, 0
label = "Normal"
color = (255, 255, 255)

# Diccionarios para llevar el registro de las manzanas malas, medias y colores por ID
estado_manzanas = {}
color_manzanas = {}

# Variables de conteo globales
contmalas = 0
contmedias = 0
cont_red_d = 0
cont_fuji = 0
cont_golden = 0
cont_granny = 0
totales = 0

ids_red_d = set()
ids_fuji = set()
ids_golden = set()
ids_granny = set()  
# Flask app setup
app = Flask(__name__)

def contar_elementos(array):
    num_unos = 0
    num_ceros = 0
    for elem in array:
        if elem:
            if elem == 1:
                num_unos += 1
            elif elem == 0:
                num_ceros += 1
    return num_unos, num_ceros

def determinar_color(region_manzana, track_id):
    if region_manzana is None or region_manzana.size == 0:
        return 'otro'

    hsv_region = cv2.cvtColor(region_manzana, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 200])
    upper_red = np.array([1, 255, 255])
    lower_green = np.array([30, 100, 100])
    upper_green = np.array([40, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_fiji = np.array([0, 100, 100])
    upper_fiji = np.array([10, 255, 255])

    mask_red = cv2.inRange(hsv_region, lower_red, upper_red)
    mask_green = cv2.inRange(hsv_region, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv_region, lower_yellow, upper_yellow)
    mask_fiji = cv2.inRange(hsv_region, lower_fiji, upper_fiji)

    avg_red = np.mean(cv2.bitwise_and(region_manzana, region_manzana, mask=mask_red), axis=(0, 1))
    avg_green = np.mean(cv2.bitwise_and(region_manzana, region_manzana, mask=mask_green), axis=(0, 1))
    avg_yellow = np.mean(cv2.bitwise_and(region_manzana, region_manzana, mask=mask_yellow), axis=(0, 1))
    avg_fiji = np.mean(cv2.bitwise_and(region_manzana, region_manzana, mask=mask_fiji), axis=(0, 1))

    if avg_red[2] > avg_red[0] and avg_red[2] > avg_red[1]:
        return 'Red D'
    elif avg_fiji[2] > avg_fiji[0] and avg_fiji[2] > avg_fiji[1]:
        return 'Fuji'
    elif avg_yellow[2] > avg_yellow[0] and avg_yellow[1] > avg_yellow[0] and avg_yellow[2] > avg_green[2]:
        return 'Golden'
    elif avg_green[1] > avg_green[0] and avg_green[1] > avg_green[2]:
        return 'Granny'
    else:
        return 'fiji'

model = YOLO("yolov8x.pt")
model2 = YOLO("bestpodridoamarillodeteccion.pt")
tracker = Sort()

# Función para controlar el Tello
def controlar_tello():
    
    try:
        print("Conexión exitosa con el dron Tello.")

        

        # Esperar unos segundos después de la conexión antes de enviar comandos
        time.sleep(5)

        # Realizar el despegue
        #tello.takeoff()
        print("Despegue exitoso.")

        # Abre el archivo en modo lectura
        with open('archivo.txt', 'r') as archivo:
            # Lee las líneas del archivo
            lineas = archivo.readlines()

            # Itera sobre las líneas y determina la acción para cada palabra clave
            for linea in lineas:
                palabras = linea.strip().split()  # Elimina espacios y divide las palabras
                for palabra in palabras:
                    if palabra == 'ADELANTE':
                        tello.send_rc_control(0, 50, 0, 0)
                        print('ADELANTE')
                    elif palabra == 'ATRAS':
                        tello.send_rc_control(0, -50, 0, 0)
                        print("ATRAS")
                    elif palabra == 'DERECHA':
                        tello.send_rc_control(50, 0, 0, 0)
                        print("DERECHA")
                    elif palabra == 'IZQUIERDA':
                        tello.send_rc_control(-50, 0, 0, 0)
                        print("IZQUIERDA")
                    elif palabra == 'ARRIBA':
                        tello.send_rc_control(0, 0, 50, 0)
                        print("ARRIBA")
                    elif palabra == 'ABAJO':
                        tello.send_rc_control(0, 0, -50, 0)
                        print("ABAJO")
                    elif palabra == 'ROTIZ':
                        tello.send_rc_control(0, 0, 0, -45)
                        print("ROTIZ")
                    elif palabra == 'ROTDER':
                        tello.send_rc_control(0, 0, 0, 45)
                        print("ROTDER")
                    elif palabra == 'ATERRIZAR':
                        tello.land()
                        print("ATERRIZAR")
                    else:
                        print("Palabra clave no válida:", palabra)

                    time.sleep(2)  # Espera después de cada comando
                    tello.send_rc_control(0, 0, 0, 0)
                    time.sleep(3)

    except Exception as e:
        print("Error:", e)

# Función para recibir y mostrar video del Tello
def recibir_video_tello():
    global contmalas, contmedias, cont_red_d, cont_fuji, cont_golden, cont_granny, totales
    tello.streamon()
    print(tello.get_battery())

    while True:
        frame = tello.get_frame_read().frame
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        results = model(frame, conf=0.3 , show=False, show_boxes=False, show_labels=False, stream=True, classes=[47])
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)

            carpeta_regiones = "regiones_manzanas"
            if not os.path.exists(carpeta_regiones):
                os.makedirs(carpeta_regiones)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                region_manzana = frame[ymin:ymax, xmin:xmax]
                ruta_archivo = os.path.join(carpeta_regiones, f"region_manzana_{track_id}.jpg")
                cv2.imwrite(ruta_archivo, region_manzana)

                color_manzana = determinar_color(region_manzana, track_id)
                color_manzanas[track_id] = color_manzana

                if region_manzana is not None and region_manzana.size != 0:
                    results2 = model2(region_manzana, show=False, show_boxes=False, show_labels=False, stream=True, classes=[0, 1], conf=0.5)
                    for result2 in results2:
                        john = result2.boxes.cls.cpu().numpy()
                        num_unos, num_ceros = contar_elementos(john)

                        if num_unos >= 1:
                            estado_manzanas[track_id] = "mala"
                        elif num_ceros >= 1:
                            estado_manzanas[track_id] = "medio"
                        print("----------------------------")
                        print(track_id)
                        
                        print("----------------------------")
                if track_id not in ids_red_d and color_manzana == "Red D":
                    color = (0, 0, 255)
                    cont_red_d += 1
                    ids_red_d.add(track_id)
                elif track_id not in ids_fuji and color_manzana == "Fuji":
                    color = (255, 0, 0)
                    cont_fuji += 1
                    ids_fuji.add(track_id)
                elif track_id not in ids_golden and color_manzana == "Golden":
                    color = (0, 255, 255)
                    cont_golden += 1
                    ids_golden.add(track_id)
                elif track_id not in ids_granny and color_manzana == "Granny":
                    color = (0, 255, 0)
                    cont_granny += 1
                    ids_granny.add(track_id)
                else:
                    color = (255, 255, 255)

                estado_manzana = estado_manzanas.get(track_id, "Desconocido")
                if estado_manzana == "mala":
                    contmalas += 1
                elif estado_manzana == "medio":
                    contmedias += 1
                else:
                    totales += 1

                label = f"ID:{track_id} {estado_manzana} {color_manzana}"


                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
                cv2.putText(frame, f"Tipo de manzana: {color_manzana} Estado:{estado_manzanas.get(track_id, 'Normal')}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
                
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(recibir_video_tello(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_program', methods=['POST'])
def start_program():
    # Crear y lanzar el hilo para controlar el Tello y recibir video
    hilo_control = threading.Thread(target=controlar_tello)
    hilo_control.start()
    
    hilo_video = threading.Thread(target=recibir_video_tello)
    hilo_video.start()
    
    return '', 200
@app.route('/stop_program', methods=['POST'])
def stop_program():
    global tello  # Hacer referencia a la variable global del dron

    # Detener la transmisión de video desde el dron
    tello.streamoff()
    contmalas = list(estado_manzanas.values()).count("mala")
    contmedias = list(estado_manzanas.values()).count("medio")
    cont_red_d = list(color_manzanas.values()).count("Red D")
    cont_fuji = list(color_manzanas.values()).count("Fuji")
    cont_golden = list(color_manzanas.values()).count("Golden")
    cont_granny = list(color_manzanas.values()).count("Granny")

    buenas=int(totales)-contmalas-contmedias
    
    print(f"Manzanas malas: {contmalas}")
    print(f"Manzanas medias: {contmedias}")
    print(f"Manzanas Red D: {cont_red_d}")
    print(f"Manzanas Fuji: {cont_fuji}")
    print(f"Manzanas Golden: {cont_golden}")
    print(f"Manzanas Granny: {cont_granny}")
    print(f"Manzanas totales: {totales}")
    print(f"Manzanas buenas: {buenas}")


    # Directorio con las imágenes
    folder_path = 'regiones_manzanas'

    # Conectar a la base de datos SQLite3 (o crearla)
    conn = sqlite3.connect('imagenes.db')
    cursor = conn.cursor()

    # Crear tabla para almacenar imágenes y contadores
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS imagenes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre
        contra
        imagen1 BLOB,
        imagen2 BLOB,
        imagen3 BLOB,
        imagen4 BLOB,
        imagen5 BLOB,
        cont_malas INTEGER,
        cont_medias INTEGER,
        cont_red_d INTEGER,
        cont_fuji INTEGER,
        cont_golden INTEGER,
        cont_granny INTEGER,
        cont_totales INTEGER,
        cont_buenas INTEGER
    )
    ''')

    # Función para convertir una imagen en un blob
    def convert_to_binary_data(filename):
        with open(filename, 'rb') as file:
            blob_data = file.read()
        return blob_data

    # Contar y procesar archivos JPG en la carpeta
    count = 0
    imagenes_blob = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            count += 1
            if count <= 5:
                filepath = os.path.join(folder_path, filename)
                blob_data = convert_to_binary_data(filepath)
                imagenes_blob.append(blob_data)
            
            # Si se han añadido 5 imágenes, insertarlas en la base de datos
            if len(imagenes_blob) == 5:
                cursor.execute('''
                INSERT INTO imagenes (imagen1, imagen2, imagen3, imagen4, imagen5, cont_malas, cont_medias, cont_red_d, cont_fuji, cont_golden, cont_granny, cont_totales,cont_buenas)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?)
                ''', (*imagenes_blob, contmalas, contmedias, cont_red_d, cont_fuji, cont_golden, cont_granny,totales,buenas))
                imagenes_blob = []

    # Insertar las imágenes restantes si hay menos de 5
    if len(imagenes_blob) > 0:
        # Rellenar con None para que siempre haya 5 columnas
        while len(imagenes_blob) < 5:
            imagenes_blob.append(None)
        cursor.execute('''
        INSERT INTO imagenes (imagen1, imagen2, imagen3, imagen4, imagen5, cont_malas, cont_medias, cont_red_d, cont_fuji, cont_golden, cont_granny,cont_totales,cont_buenas)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?)
        ''', (*imagenes_blob, contmalas, contmedias, cont_red_d, cont_fuji, cont_golden, cont_granny,totales,buenas))

    # Confirmar cambios en la base de datos
    conn.commit()

    # Eliminar archivos JPG de la carpeta
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            os.remove(os.path.join(folder_path, filename))

    # Cerrar la conexión a la base de datos
    conn.close()

    print(f'Se han procesado y subido a la base de datos las primeras 5 imágenes JPG junto con los contadores. Todas las imágenes han sido eliminadas de la carpeta.')

    #----------------------------
    # Detener cualquier hilo en ejecución
    for thread in threading.enumerate():
        if thread != threading.current_thread():  # Evitar detener el hilo actual de Flask
            thread.join()

    return '', 200

if __name__ == "__main__":
    # Inicializar el Tello pero sin conectar automáticamente
    tello = Tello()
    tello.connect()
    
    # Iniciar la aplicación Flask sin conectar automáticamente al dron
    app.run(host='0.0.0.0', port=5000)
