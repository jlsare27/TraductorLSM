import cv2
import mediapipe as mp
import json
import numpy as np
from sklearn.svm import SVC
from base_de_datos.conexion import conectar
from captura_señas.utilidades import normalizar_keypoints, calcular_angulos, calcular_distancias

# Inicializar MediaPipe para Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def cargar_gestures():
    conexion = conectar()
    gestures_data = []

    if conexion:
        cursor = conexion.cursor()
        consulta = "SELECT id, sign_name, angles, keypoints, distances FROM gestures"
        cursor.execute(consulta)
        gestures = cursor.fetchall()

        for gesture in gestures:
            gesture_id = gesture[0]
            nombre_seña = gesture[1]
            angles_json = gesture[2]
            keypoints_json = gesture[3]
            distances_json = gesture[4]

            if angles_json and keypoints_json and distances_json:
                try:
                    angles = json.loads(angles_json)
                    keypoints = json.loads(keypoints_json)
                    distances = json.loads(distances_json)
                    gestures_data.append({
                        'id': gesture_id,
                        'sign_name': nombre_seña,
                        'angles': angles,
                        'keypoints': keypoints,
                        'distances': distances
                    })
                except json.JSONDecodeError as e:
                    print(f"Error al decodificar JSON para la seña '{nombre_seña}': {e}")
            else:
                print(f"Advertencia: La seña '{nombre_seña}' tiene campos vacíos.")

        cursor.close()
        conexion.close()

    return gestures_data

def entrenar_clasificador(gestures_data):
    X = []
    y = []

    for gesture in gestures_data:
        nombre_seña = gesture['sign_name']
        keypoints = gesture['keypoints']
        angles = gesture['angles']

        # Normalizar los keypoints
        keypoints_normalizados = normalizar_keypoints(keypoints)
        # Aplanar los keypoints para que estén en una sola lista de valores
        keypoints_flat = [value for kp in keypoints_normalizados for value in kp.values()]

        # Calcular las distancias entre puntos clave
        distancias = calcular_distancias(keypoints_normalizados)  # Ahora es una lista, no un diccionario

        # Concatenar keypoints, ángulos y distancias en una sola entrada
        entrada = keypoints_flat + angles + distancias

        # Agregar la entrada y su respectiva etiqueta (nombre de la seña)
        X.append(entrada)
        y.append(nombre_seña)

    # Verificar que haya datos suficientes para entrenar el modelo
    if len(X) == 0:
        raise ValueError("No hay datos disponibles para entrenar el modelo SVM.")

    # Crear y entrenar el clasificador SVM
    svm = SVC(kernel='linear')  # Puedes ajustar el kernel según las necesidades (e.g., 'rbf', 'poly')
    svm.fit(X, y)

    return svm

def reconocer_seña(svm, keypoints_actuales):
    keypoints_normalizados = normalizar_keypoints(keypoints_actuales)
    angles_actuales = calcular_angulos(keypoints_normalizados)
    distancias_actuales = calcular_distancias(keypoints_normalizados)

    keypoints_flat = [value for kp in keypoints_normalizados for value in kp.values()]
    entrada = keypoints_flat + angles_actuales + distancias_actuales

    try:
        seña_reconocida = svm.predict([entrada])
        return seña_reconocida[0]
    except Exception as e:
        print(f"Error en reconocimiento: {e}")
        return "Seña no reconocida"

def reconocer_señas_en_tiempo_real():
    gestures_data = cargar_gestures()
    
    if not gestures_data:
        print("No se encontraron señas en la base de datos.")
        return

    svm = entrenar_clasificador(gestures_data)
    print("Modelo SVM entrenado exitosamente.")

    cap = cv2.VideoCapture(0)

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

                    if keypoints_actuales:
                        seña_reconocida = reconocer_seña(svm, keypoints_actuales)
                        if seña_reconocida:
                            cv2.putText(
                                image_bgr, 
                                seña_reconocida, 
                                (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (255, 0, 0), 
                                2, 
                                cv2.LINE_AA
                            )

            cv2.imshow("Reconocimiento de Señas", image_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()