import numpy as np
from sympy import symbols, lambdify, expand
import matplotlib.pyplot as plt

# Función para construir la tabla de diferencias divididas
def diferencias_divididas(X, Y):
    n = len(X)
    coef = np.zeros([n, n])
    coef[:, 0] = Y  # La primera columna son los valores Y

    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (X[i + j] - X[i])

    print("\nTabla de diferencias divididas:")
    for i in range(n):
        for j in range(n - i):
            print(f"f[{i},{j}] = {coef[i, j]}")

    return coef[0]

# Función para construir el polinomio de Newton sin expandir
def construir_polinomio_newton(X, coefs):
    x = symbols('x')
    n = len(coefs)
    polinomio = f"{coefs[0]}"
    print("\nConstrucción del polinomio de Newton (sin expandir):")
    print(f"P(x) = {coefs[0]}")

    for i in range(1, n):
        term = f" + {coefs[i]}"
        for j in range(i):
            term += f" * (x - {X[j]})"
        polinomio += term
        print(f"P(x) = {polinomio}")

    return polinomio

# Función para expandir el polinomio de Newton
def expandir_polinomio(X, coefs):
    x = symbols('x')
    n = len(coefs)
    polinomio = coefs[0]
    producto = 1

    for i in range(1, n):
        producto *= (x - X[i - 1])
        polinomio += coefs[i] * producto

    polinomio_expandido = expand(polinomio)
    return polinomio_expandido

# Función para evaluar el polinomio en puntos X
def evaluar_polinomio(polinomio, X):
    x = symbols('x')
    polinomio_lambdified = lambdify(x, polinomio)
    return [polinomio_lambdified(val) for val in X]

# Función para graficar el polinomio
def graficar_polinomio_newton(X, Y, polinomio):
    x = symbols('x')
    polinomio_lambdified = lambdify(x, polinomio)
    X_curva = np.linspace(min(X) - 1, max(X) + 1, 500)
    Y_curva = polinomio_lambdified(X_curva)

    plt.plot(X_curva, Y_curva, label='Polinomio Interpolante', color='blue')
    plt.scatter(X, Y, color='red', label='Puntos Originales', zorder=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Interpolación con el Polinomio de Newton')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para calcular L_i en Lagrange
import numpy as np
from sympy import symbols, lambdify, expand
import matplotlib.pyplot as plt


# Función para calcular L_i en Lagrange
def calcular_Li(xi, i, x):
    n = len(xi)
    Li_num = 1
    Li_den = 1
    Li_exp = f"L_{i}(x) = (x"

    for j in range(n):
        if i != j:
            Li_num *= (x - xi[j])
            Li_den *= (xi[i] - xi[j])
            Li_exp += f" - {xi[j]:.4f})"

    Li_exp += f" / ({Li_den:.4f})"
    print(f"i={i}\n{Li_exp}")
    return Li_num / Li_den


# Función para ensamblar el polinomio de Lagrange
def lagrange_polinomio(xi, yi):
    x = symbols('x')
    n = len(xi)
    polinomio = 0
    print("Teniendo los polinomios L_i(x), podemos ensamblar el polinomio de Lagrange:\n")

    for i in range(n):
        Li = calcular_Li(xi, i, x)
        print(f"L_{i}(x) = {Li}\n")
        polinomio += yi[i] * Li
        print(f"f_{i}(x) = {yi[i]:.4f} * L_{i}(x)\n")

    print("\nEl polinomio de Lagrange final es:")
    polinomio = expand(polinomio)  # Expande el polinomio para mostrar su forma final
    print(polinomio)
    return polinomio


# Función para graficar el polinomio de Lagrange
def graficar_polinomio_lagrange(X, Y, polinomio):
    x = symbols('x')
    polinomio_lambdified = lambdify(x, polinomio)

    # Generar puntos para la curva del polinomio
    X_curva = np.linspace(min(X) - 1, max(X) + 1, 500)
    Y_curva = polinomio_lambdified(X_curva)

    # Graficar puntos y polinomio
    plt.plot(X_curva, Y_curva, label='Polinomio de Lagrange', color='purple')
    plt.scatter(X, Y, color='blue', label='Puntos Originales')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Interpolación con el Polinomio de Lagrange')
    plt.legend()
    plt.grid(True)
    plt.show()

def calcular_trazadora_cubica(X, Y):
    """
    Calcula los coeficientes de las trazadoras cúbicas para un conjunto de puntos X, Y,
    mostrando paso a paso todo el procedimiento.
    """
    n = len(X) - 1  # Número de intervalos
    h = np.diff(X)  # Distancias entre los puntos en X

    print("\nPASO 1: Cálculo de los h_j")
    for j in range(n):
        print(f"h_{j} = X[{j+1}] - X[{j}] = {X[j+1]} - {X[j]} = {h[j]}")

    # Cálculo de alfa
    alfa = np.zeros(n)
    print("\nPASO 2: Cálculo de alfa_j")
    for i in range(1, n):
        alfa[i] = (3/h[i]) * (Y[i+1] - Y[i]) - (3/h[i-1]) * (Y[i] - Y[i-1])
        print(f"alfa_{i} = 3/h[{i}] * (Y[{i+1}] - Y[{i}]) - 3/h[{i-1}] * (Y[{i}] - Y[{i-1}]) = {alfa[i]}")

    # Inicialización de los arrays l, mu, z
    l = np.ones(n+1)
    mu = np.zeros(n+1)
    z = np.zeros(n+1)

    # Paso 4
    l[0] = 1
    mu[0] = 0
    z[0] = 0
    print(f"\nPASO 3 y 4: Inicialización: l[0] = {l[0]}, mu[0] = {mu[0]}, z[0] = {z[0]}")

    # Resolución del sistema tridiagonal
    print("\nPASO 5: Resolución del sistema tridiagonal")
    for i in range(1, n):
        l[i] = 2 * (X[i+1] - X[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alfa[i] - h[i-1] * z[i-1]) / l[i]
        print(f"l_{i} = 2 * (X[{i+1}] - X[{i-1}]) - h[{i-1}] * mu[{i-1}] = {l[i]}")
        print(f"mu_{i} = h[{i}] / l[{i}] = {mu[i]}")
        print(f"z_{i} = (alfa_{i} - h[{i-1}] * z[{i-1}]) / l[{i}] = {z[i]}")

    l[n] = 1
    z[n] = 0
    print(f"l[{n}] = {l[n]}, z[{n}] = {z[n]}")

    # Inicialización de los coeficientes a, b, c, d
    a = np.copy(Y)
    b = np.zeros(n)
    c = np.zeros(n+1)
    d = np.zeros(n)

    # Paso 7
    print("\nPASO 6: Cálculo de los coeficientes c, b, y d")
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (Y[j+1] - Y[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])
        print(f"c_{j} = z_{j} - mu_{j} * c_{j+1} = {c[j]}")
        print(f"b_{j} = (Y[{j+1}] - Y[{j}]) / h[{j}] - h[{j}] * (c_{j+1} + 2 * c_{j}) / 3 = {b[j]}")
        print(f"d_{j} = (c_{j+1} - c_{j}) / (3 * h[{j}]) = {d[j]}")

    return a, b, c, d

def imprimir_polinomios(a, b, c, d, X):
    """
    Imprime los polinomios cúbicos resultantes para cada intervalo [X_i, X_{i+1}].
    """
    n = len(a) - 1
    print("\nPolinomios resultantes:")
    for i in range(n):
        print(f"S_{i}(x) = {a[i]:.4f} + {b[i]:.4f}(x - {X[i]:.4f}) + {c[i]:.4f}(x - {X[i]:.4f})^2 + {d[i]:.4f}(x - {X[i]:.4f})^3, para x en [{X[i]:.4f}, {X[i+1]:.4f}]")

def trazadora_polynomial(a, b, c, d, X, x):
    """
    Evalúa la trazadora cúbica en un valor x dado.
    """
    n = len(a) - 1
    for i in range(n):
        if X[i] <= x <= X[i+1]:
            dx = x - X[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    return None  # Devuelve None si x está fuera del intervalo

def graficar_trazadora_cubica(X, Y, a, b, c, d):
    """
    Grafica las trazadoras cúbicas y los puntos originales.
    """
    X_curva = np.linspace(min(X), max(X), 500)
    Y_curva = [trazadora_polynomial(a, b, c, d, X, x) for x in X_curva]

    # Graficar puntos y trazadora cúbica
    plt.plot(X_curva, Y_curva, label='Trazadora Cúbica', color='green')
    plt.scatter(X, Y, color='red', label='Puntos Originales')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Interpolación con Trazadoras Cúbicas')
    plt.legend()
    plt.grid(True)
    plt.show()


# Menú principal
def menu():
    while True:
        print("\n--- Menú de interpolación ---")
        print("1. Interpolación de Newton (Diferencias Divididas)")
        print("2. Interpolación de Lagrange")
        print("3. trazadoras cubicas")
        print("4. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            X = list(map(float, input("Ingrese los valores de X separados por espacios: ").split()))
            Y = list(map(float, input("Ingrese los valores de Y separados por espacios: ").split()))

            coefs = diferencias_divididas(X, Y)
            polinomio_sin_expandir = construir_polinomio_newton(X, coefs)
            polinomio_expandido = expandir_polinomio(X, coefs)

            print(f"\nEl polinomio de Newton expandido es: {polinomio_expandido}")
            Y_evaluado = evaluar_polinomio(polinomio_expandido, X)
            print("\nVerificación: Evaluación del polinomio en los puntos originales X:")
            for xi, yi, y_eval in zip(X, Y, Y_evaluado):
                print(f"P({xi}) = {y_eval} (Valor original: {yi})")

            graficar_polinomio_newton(X, Y, polinomio_expandido)

        elif opcion == "2":
            X = list(map(float, input("Ingrese los valores de X separados por espacios: ").split()))
            Y = list(map(float, input("Ingrese los valores de Y separados por espacios: ").split()))

            polinomio = lagrange_polinomio(X, Y)  # Calcula el polinomio completo sin pedir un valor de x
            graficar_polinomio_lagrange(X, Y, polinomio)

        elif opcion == "3":
            # Ingresar valores de X y Y
            X = list(map(float, input("Ingrese los valores de X separados por espacios: ").split()))
            Y = list(map(float, input("Ingrese los valores de Y separados por espacios: ").split()))

            # Calcular trazadora cúbica
            a, b, c, d = calcular_trazadora_cubica(X, Y)

            # Imprimir polinomios resultantes
            imprimir_polinomios(a, b, c, d, X)

            # Graficar la trazadora cúbica
            graficar_trazadora_cubica(X, Y, a, b, c, d)

        elif opcion == "4":
            print("Saliendo del programa...")
            break

        else:
            print("Opción no válida. Intente nuevamente.")

# Ejecutar el menú
menu()
