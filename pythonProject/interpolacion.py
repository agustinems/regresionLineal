import numpy as np
from sympy import symbols, lambdify, expand
import matplotlib.pyplot as plt


def diferencias_divididas(X, Y):
    """
    Construcción de la tabla de diferencias divididas y cálculo del polinomio de Newton.
    """
    n = len(X)
    coef = np.zeros([n, n])
    coef[:, 0] = Y  # Primera columna son los valores Y

    # Construcción de la tabla de diferencias divididas
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (X[i + j] - X[i])

    # Mostrar la tabla de diferencias divididas
    print("\nTabla de diferencias divididas:")
    for i in range(n):
        for j in range(n - i):
            print(f"f[{i},{j}] = {coef[i, j]}")

    return coef[0]  # Retorna los coeficientes del polinomio


def construir_polinomio_newton(X, coefs):
    """
    Construye el polinomio de Newton sin expandirlo, mostrando el procedimiento paso a paso.
    """
    x = symbols('x')
    n = len(coefs)
    polinomio = f"{coefs[0]}"
    producto = 1
    print("\nConstrucción del polinomio de Newton (sin expandir):")
    print(f"P(x) = {coefs[0]}")

    # Construcción del polinomio paso a paso
    for i in range(1, n):
        term = f" + {coefs[i]}"
        for j in range(i):
            term += f" * (x - {X[j]})"
        polinomio += term
        print(f"P(x) = {polinomio}")

    return polinomio


def expandir_polinomio(X, coefs):
    """
    Expande el polinomio de Newton para obtener la forma simplificada.
    """
    x = symbols('x')
    n = len(coefs)
    polinomio = coefs[0]
    producto = 1

    # Construcción del polinomio con términos sucesivos
    for i in range(1, n):
        producto *= (x - X[i - 1])
        polinomio += coefs[i] * producto

    # Expandir y simplificar el polinomio
    polinomio_expandido = expand(polinomio)

    return polinomio_expandido


def evaluar_polinomio(polinomio, X):
    """
    Evalúa el polinomio expandido en los puntos X proporcionados.
    """
    x = symbols('x')
    polinomio_lambdified = lambdify(x, polinomio)  # Convertir el polinomio simbólico a una función evaluable
    return [polinomio_lambdified(val) for val in X]


def graficar_polinomio(X, Y, polinomio):
    """
    Grafica los puntos originales y la curva del polinomio de interpolación.
    """
    x = symbols('x')
    polinomio_lambdified = lambdify(x, polinomio)

    # Generar puntos para la curva del polinomio
    X_curva = np.linspace(min(X) - 1, max(X) + 1, 500)
    Y_curva = polinomio_lambdified(X_curva)

    # Graficar puntos y polinomio
    plt.plot(X_curva, Y_curva, label='Polinomio Interpolante', color='blue')
    plt.scatter(X, Y, color='red', label='Puntos Originales', zorder=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Interpolación con el Polinomio de Newton')
    plt.legend()
    plt.grid(True)
    plt.show()


def menu():
    """
    Función que muestra el menú de opciones al usuario.
    """
    while True:
        print("\n--- Menú de interpolación ---")
        print("1. Interpolación de Newton (Diferencias Divididas)")
        print("2. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            # Ingresar valores de X y Y
            X = list(map(float, input("Ingrese los valores de X separados por espacios: ").split()))
            Y = list(map(float, input("Ingrese los valores de Y separados por espacios: ").split()))

            # Cálculo de las diferencias divididas
            coefs = diferencias_divididas(X, Y)

            # Mostrar el polinomio sin expandir (procedimiento)
            polinomio_sin_expandir = construir_polinomio_newton(X, coefs)

            # Expansión del polinomio
            polinomio_expandido = expandir_polinomio(X, coefs)

            # Mostrar polinomio reducido
            print(f"\nEl polinomio de Newton reducido es: {polinomio_expandido}")

            # Evaluar el polinomio en los puntos dados
            Y_evaluado = evaluar_polinomio(polinomio_expandido, X)
            print("\nVerificación: Evaluación del polinomio en los puntos originales X:")
            for xi, yi, y_eval in zip(X, Y, Y_evaluado):
                print(f"P({xi}) = {y_eval} (Valor original: {yi})")

            # Graficar el polinomio y los puntos originales
            graficar_polinomio(X, Y, polinomio_expandido)

        elif opcion == "2":
            print("Saliendo del programa...")
            break

        else:
            print("Opción no válida. Intente nuevamente.")


# Ejecutar el menú
menu()
