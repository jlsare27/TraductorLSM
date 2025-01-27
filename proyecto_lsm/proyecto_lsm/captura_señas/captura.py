import cv2
import os
import mediapipe as mp
import json
from base_de_datos.conexion import conectar
from captura_señas.utilidades import normalizar_keypoints, calcular_angulos, calcular_distancias

# Inicializar MediaPipe para Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def solicitar_datos_seña():
    etiqueta = input("Introduce el nombre de la seña (ej. 'Por favor'): ")
    categoria = input("Introduce la categoría de la seña (ej. 'Gestos comunes'): ")
    return etiqueta, categoria

def registrar_seña():
    etiqueta, categoria = solicitar_datos_seña()

    carpeta = f"./datasets/{etiqueta}"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    cap = cv2.VideoCapture(0)
    capturando = False  # Bandera para iniciar/detener la captura

    gesture_id = obtener_gesture_id(etiqueta)
    if gesture_id is None:
        gesture_id = guardar_gesture(etiqueta, categoria)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            results = hands.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    keypoints_actuales = []
                    for landmark in hand_landmarks.landmark:
                        keypoints_actuales.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })

                    if not keypoints_actuales:
                        print("No se detectaron keypoints.")
                        continue

                    keypoints_normalizados = normalizar_keypoints(keypoints_actuales)
                    
                    angulos = calcular_angulos(keypoints_normalizados)
                    distancias = calcular_distancias(keypoints_normalizados)

                    keypoints_json = json.dumps(keypoints_normalizados)
                    angulos_json = json.dumps(angulos)
                    distancias_json = json.dumps(distancias)

                    h, w, _ = frame.shape
                    x_min = min([int(kp['x'] * w) for kp in keypoints_actuales])
                    x_max = max([int(kp['x'] * w) for kp in keypoints_actuales])
                    y_min = min([int(kp['y'] * h) for kp in keypoints_actuales])
                    y_max = max([int(kp['y'] * h) for kp in keypoints_actuales])

                    # Aumentar el margen alrededor de la mano
                    margen = 30  # Ajusta el valor para el margen deseado
                    x_min = max(x_min - margen, 0)
                    y_min = max(y_min - margen, 0)
                    x_max = min(x_max + margen, w)
                    y_max = min(y_max + margen, h)

                    recorte_manos = image_bgr[y_min:y_max, x_min:x_max]

                    if capturando:
                        ruta_imagen = f"{carpeta}/imagen_{gesture_id}.jpg"
                        cv2.imwrite(ruta_imagen, recorte_manos)
                        print(f"Imagen recortada guardada en: {ruta_imagen}")

                        guardar_en_base_datos(gesture_id, ruta_imagen)
                        guardar_keypoints(gesture_id, keypoints_json, angulos_json, distancias_json)

                        print("Captura completada.")
                        capturando = False

            cv2.imshow("Capturando seña fija - Presiona 's' para capturar, 'q' para salir", image_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                capturando = True
                print("Iniciando captura...")

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def guardar_gesture(etiqueta, categoria):
    conexion = conectar()
    if conexion:
        cursor = conexion.cursor()
        consulta = "INSERT INTO gestures (sign_name, category) VALUES (%s, %s)"
        cursor.execute(consulta, (etiqueta, categoria))
        conexion.commit()
        gesture_id = cursor.lastrowid
        cursor.close()
        conexion.close()
        return gesture_id

def obtener_gesture_id(etiqueta):
    conexion = conectar()
    if conexion:
        cursor = conexion.cursor()
        consulta = "SELECT id FROM gestures WHERE sign_name = %s"
        cursor.execute(consulta, (etiqueta,))
        result = cursor.fetchone()
        cursor.close()
        conexion.close()
        if result:
            return result[0]
        return None

def guardar_en_base_datos(gesture_id, ruta_imagen):
    conexion = conectar()
    if conexion:
        cursor = conexion.cursor()
        consulta = "INSERT INTO imagenes_señas (gesture_id, ruta_imagen) VALUES (%s, %s)"
        cursor.execute(consulta, (gesture_id, ruta_imagen))
        conexion.commit()
        print("Imagen registrada en la base de datos")
        cursor.close()
        conexion.close()

def guardar_keypoints(gesture_id, keypoints_json, angulos_json, distancias_json):
    conexion = conectar()
    if conexion:
        cursor = conexion.cursor()
        consulta = "UPDATE gestures SET keypoints = %s, angles = %s, distances = %s WHERE id = %s"
        cursor.execute(consulta, (keypoints_json, angulos_json, distancias_json, gesture_id))
        conexion.commit()
        print("Keypoints, ángulos y distancias guardados en la base de datos")
        cursor.close()
        conexion.close()