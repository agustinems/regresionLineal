import numpy as np
import matplotlib.pyplot as plt

class SplineCubico:
    def __init__(self, a, b, c, d, x):
        self.a = a  # Coeficiente a (término constante)
        self.b = b  # Coeficiente b (término lineal)
        self.c = c  # Coeficiente c (término cuadrático)
        self.d = d  # Coeficiente d (término cúbico)
        self.x = x  # Coordenada x al inicio del intervalo

def spline_cubico_natural(x, y):
    n = len(x) - 1
    h = [x[i+1] - x[i] for i in range(n)]
    alpha = [0]*n
    for i in range(1, n):
        alpha[i] = (3/h[i])*(y[i+1] - y[i]) - (3/h[i-1])*(y[i] - y[i-1])

    l = [0]*(n+1)
    mu = [0]*n
    z = [0]*(n+1)
    l[0] = 1.0
    mu[0] = z[0] = 0.0

    for i in range(1, n):
        l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]

    l[n] = 1.0
    z[n] = 0.0

    c = [0]*(n+1)
    b = [0]*n
    d = [0]*n
    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1] - y[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3
        d[j] = (c[j+1] - c[j])/(3*h[j])

    splines = []
    for i in range(n):
        s = SplineCubico(y[i], b[i], c[i], d[i], x[i])
        splines.append(s)
    return splines

def evaluar_spline(splines, x_eval):
    for i, s in enumerate(splines):
        if x_eval >= s.x and (i == len(splines) - 1 or x_eval < splines[i+1].x):
            dx = x_eval - s.x
            return s.a + s.b * dx + s.c * dx**2 + s.d * dx**3
    raise ValueError("El valor x está fuera del rango de interpolación.")

def imprimir_polinomios_reducidos(splines, x_max):
    for i, s in enumerate(splines):
        a, b, c, d, x0 = s.a, s.b, s.c, s.d, s.x
        a_red = a - b*x0 + c*x0**2 - d*x0**3
        b_red = b - 2*c*x0 + 3*d*x0**2
        c_red = c - 3*d*x0
        d_red = d

        intervalo_fin = splines[i+1].x if i < len(splines) - 1 else x_max
        print(f"Polinomio {i} en el intervalo [{x0}, {intervalo_fin}]:")
        print(f"S_{i}(x) = {a_red:.4f} + {b_red:.4f} * x + {c_red:.4f} * x^2 + {d_red:.4f} * x^3\n")

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
    plt.plot(x_plot, y_plot, label='Spline Cúbico')
    plt.scatter(x_puntos, y_puntos, color='red', label='Puntos de datos')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación con Splines Cúbicos Naturales')
    plt.show()

def main():
    print("-"*60)
    print("Interpolación con splines cúbicos naturales")
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
    x_max = max(x)

    splines = spline_cubico_natural(x, y)
    imprimir_polinomios_reducidos(splines, x_max)

    # Graficar los splines sin solicitar un valor de x
    graficar_splines(splines, x, y)

if __name__ == "__main__":
    main()
