import mysql.connector

def conectar():
    try:
        conexion = mysql.connector.connect(
            host="localhost",
            user="root",  # Cambia a tu usuario de MySQL
            password="",  # Cambia a tu contraseña de MySQL
            database="datasetsLSM"  # Base de datos creada en XAMPP
        )
        if conexion.is_connected():
            print("Conexión exitosa a la base de datos")
        return conexion
    except mysql.connector.Error as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None
