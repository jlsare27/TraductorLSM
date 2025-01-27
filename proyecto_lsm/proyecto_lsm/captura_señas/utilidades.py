import numpy as np
import math

def normalizar_keypoints(keypoints):
    """
    Normaliza los keypoints en relación al tamaño de la mano y el punto base (muñeca).
    """
    # Punto base es la muñeca (primer keypoint en el array)
    base_x = keypoints[0]['x']
    base_y = keypoints[0]['y']
    base_z = keypoints[0]['z']

    # Normalizar las coordenadas de cada keypoint respecto a la muñeca
    keypoints_normalizados = []
    for kp in keypoints:
        normalizado = {
            'x': kp['x'] - base_x,
            'y': kp['y'] - base_y,
            'z': kp['z'] - base_z
        }
        keypoints_normalizados.append(normalizado)

    return keypoints_normalizados

def calcular_angulos(keypoints):
    """
    Calcula los ángulos entre los dedos usando los keypoints normalizados.
    Retorna una lista de ángulos para cada dedo.
    """
    def vector_entre_puntos(p1, p2):
        return np.array([
            p2['x'] - p1['x'],
            p2['y'] - p1['y'],
            p2['z'] - p1['z']
        ])
    
    def angulo_entre_vectores(v1, v2):
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    angulos = []
    
    # Índices de los puntos claves para calcular los ángulos de los dedos
    dedos_indices = {
        'pulgar': [1, 2, 3, 4],
        'indice': [5, 6, 7, 8],
        'medio': [9, 10, 11, 12],
        'anular': [13, 14, 15, 16],
        'meñique': [17, 18, 19, 20]
    }
    
    for indices in dedos_indices.values():
        # Vectores entre las falanges del dedo
        vector1 = vector_entre_puntos(keypoints[indices[0]], keypoints[indices[1]])
        vector2 = vector_entre_puntos(keypoints[indices[1]], keypoints[indices[2]])
        vector3 = vector_entre_puntos(keypoints[indices[2]], keypoints[indices[3]])
        
        # Ángulos entre los vectores de las falanges
        angulo1 = angulo_entre_vectores(vector1, vector2)
        angulo2 = angulo_entre_vectores(vector2, vector3)
        
        # Agregar los ángulos a la lista
        angulos.extend([angulo1, angulo2])
    
    return angulos  # Retorna una lista de valores numéricos


def calcular_distancias(keypoints):
    """
    Calcula las distancias entre puntos clave de la mano.
    Retorna una lista de distancias entre algunos puntos relevantes.
    """
    def distancia_entre_puntos(p1, p2):
        return math.sqrt(
            (p2['x'] - p1['x']) ** 2 +
            (p2['y'] - p1['y']) ** 2 +
            (p2['z'] - p1['z']) ** 2
        )

    distancias = [
        # Distancias entre algunos puntos clave, ajusta según sea necesario
        distancia_entre_puntos(keypoints[0], keypoints[4]),   # muñeca_pulgar
        distancia_entre_puntos(keypoints[0], keypoints[8]),   # muñeca_indice
        distancia_entre_puntos(keypoints[0], keypoints[12]),  # muñeca_medio
        distancia_entre_puntos(keypoints[0], keypoints[16]),  # muñeca_anular
        distancia_entre_puntos(keypoints[0], keypoints[20]),  # muñeca_meñique
        distancia_entre_puntos(keypoints[4], keypoints[8]),   # pulgar_indice
        distancia_entre_puntos(keypoints[8], keypoints[12]),  # indice_medio
    ]

    return distancias  # Retorna una lista de distancias entre puntos clave