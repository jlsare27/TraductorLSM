from captura_señas.captura import registrar_seña
from captura_señas.reconocimiento import reconocer_señas_en_tiempo_real

def mostrar_menu():
    print("\nBienvenido a la aplicación de lenguaje de señas")
    print("1. Registrar una nueva seña")
    print("2. Reconocer señas en tiempo real")
    print("3. Salir")

def main():
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ")

        if opcion == '1':
            registrar_seña()
        elif opcion == '2':
            reconocer_señas_en_tiempo_real()
        elif opcion == '3':
            print("Saliendo de la aplicación.")
            break
        else:
            print("Opción no válida. Inténtalo de nuevo.")

if __name__ == "__main__":
    main()
