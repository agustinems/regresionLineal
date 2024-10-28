#en el punto 1 fp0 =1 y fp1 =10

import numpy as np
import matplotlib.pyplot as plt

class CubicSpline:
    def __init__(self, a, b, c, d, x):
        self.a = a  # Coeficiente a (término constante)
        self.b = b  # Coeficiente b (término lineal)
        self.c = c  # Coeficiente c (término cuadrático)
        self.d = d  # Coeficiente d (término cúbico)
        self.x = x  # Coordenada x al inicio del intervalo

def cubic_spline_conditioned(x, y, fp0, fpn):
    n = len(x) - 1
    h = [x[i+1] - x[i] for i in range(n)]
    alpha = [0]*(n + 1)

    # Paso 2: Calcular valores alpha incluyendo las condiciones de contorno
    alpha[0] = 3 * ((y[1] - y[0]) / h[0] - fp0)
    for i in range(1, n):
        alpha[i] = (3 / h[i]) * (y[i+1] - y[i]) - (3 / h[i-1]) * (y[i] - y[i-1])
    alpha[n] = 3 * (fpn - (y[n] - y[n-1]) / h[n-1])

    # Paso 3: Resolver el sistema tridiagonal
    l = [0]*(n+1)
    mu = [0]*(n)
    z = [0]*(n+1)
    l[0] = 2 * h[0]
    mu[0] = 0.5
    z[0] = alpha[0] / l[0]

    for i in range(1, n):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]

    l[n] = h[n-1] * (2 - mu[n-1])
    z[n] = (alpha[n] - h[n-1]*z[n-1]) / l[n]

    c = [0]*(n+1)
    b = [0]*n
    d = [0]*n
    c[n] = z[n]
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1] - y[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3
        d[j] = (c[j+1] - c[j]) / (3*h[j])

    # Paso 7: Construir los coeficientes del spline
    splines = []
    for i in range(n):
        s = CubicSpline(y[i], b[i], c[i], d[i], x[i])
        splines.append(s)
    return splines

def imprimir_polinomios_reducidos(splines, x):
    for i, s in enumerate(splines):
        a = s.a
        b = s.b
        c = s.c
        d = s.d
        x0 = s.x

        # Expandir el polinomio S_i(x) = a + b*(x - x0) + c*(x - x0)^2 + d*(x - x0)^3
        a_reducido = a - b * x0 + c * x0**2 - d * x0**3
        b_reducido = b - 2 * c * x0 + 3 * d * x0**2
        c_reducido = c - 3 * d * x0
        d_reducido = d

        print(f"Polinomio {i} en el intervalo [{x[i]}, {x[i+1]}]:")
        print(f"S_{i}(x) = {a_reducido:.4f} + {b_reducido:.4f} * x + {c_reducido:.4f} * x^2 + {d_reducido:.4f} * x^3\n")

def graficar_splines(splines, x_puntos, y_puntos):
    x_plot = []
    y_plot = []
    for i, s in enumerate(splines):
        x_start = s.x
        x_end = splines[i+1].x if i < len(splines) - 1 else max(x_puntos)
        x_intervalo = np.linspace(x_start, x_end, 100)
        dx = x_intervalo - s.x
        y_intervalo = s.a + s.b * dx + s.c * dx**2 + s.d * dx**3
        x_plot.extend(x_intervalo)
        y_plot.extend(y_intervalo)
    plt.plot(x_plot, y_plot, label='Spline Cúbico Condicionado')
    plt.scatter(x_puntos, y_puntos, color='red', label='Puntos de datos')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación con Splines Cúbicos Condicionados')
    plt.show()

def main():
    print("-"*60)
    print("Interpolación con splines cúbicos condicionados")
    print("-"*60)
    x_input = input("Ingrese los valores de x separados por espacios: ")
    x = list(map(float, x_input.strip().split()))
    y_input = input("Ingrese los valores de y separados por espacios: ")
    y = list(map(float, y_input.strip().split()))

    if len(x) != len(y):
        print("Los vectores x e y deben tener la misma longitud.")
        return

    # Ordenar los puntos por x
    pares_ordenados = sorted(zip(x, y))
    x, y = zip(*pares_ordenados)
    x, y = list(x), list(y)

    # Solicitar las derivadas inicial y final
    fp0 = float(input("Ingrese el valor de la derivada en el punto inicial (fp0): "))
    fpn = float(input("Ingrese el valor de la derivada en el punto final (fpn): "))

    splines = cubic_spline_conditioned(x, y, fp0, fpn)
    imprimir_polinomios_reducidos(splines, x)

    # Graficar los splines sin solicitar un valor de x
    graficar_splines(splines, x, y)

if __name__ == "__main__":
    main()
