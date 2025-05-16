# main.py
import os

while True:
    print("\n=== ENTRENAMIENTO RED NEURONAL ===")
    print("Elegí el optimizador para entrenar la red neuronal:")
    print("1. SGD (por defecto)")
    print("2. RMSProp")
    print("3. Adam")
    print("4. SGD con Momentum")
    print("0. Salir")

    opcion = input("Ingresá el número de la opción deseada (Enter para 1): ").strip()

    if opcion == "0":
        print("Saliendo del programa.")
        break

    if opcion == "":
        opcion = 1
        print("No se seleccionó optimizador, se usará SGD (1) por defecto.")
    else:
        try:
            opcion = int(opcion)
            if opcion not in [1, 2, 3, 4]:
                raise ValueError
        except ValueError:
            print("❌ Opción inválida. Debe ser 1, 2, 3, 4 o 0.")
            continue

    epocas = input("¿Cuántas épocas de entrenamiento? (Enter para 20): ").strip()
    epocas = int(epocas) if epocas else 20

    step_size = input("¿Step size (tamaño de paso)? (Enter para 0.01): ").strip()
    step_size = float(step_size) if step_size else 0.01

    alpha = input("¿Alpha (solo para Adam)? (Enter para 0.02): ").strip()
    alpha = float(alpha) if alpha else 0.02

    # Ejecutar el entrenamiento
    os.system(f"python simple_nn_2d.py {opcion} {epocas} {step_size} {alpha}")

    input("\n✅ Entrenamiento finalizado. Presioná Enter para volver al menú...")