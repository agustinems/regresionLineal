#en el ejercicio 1 para cubica condicionada fp0 =1 fp1 =10
import numpy as np
import matplotlib.pyplot as plt

# Funciones para Interpolación de Newton
def calcular_diferencias_divididas(X, Y):
    n = len(X)
    dif = np.zeros((n, n))
    dif[:,0] = Y  # Primera columna es Y

    for i in range(1, n):
        for j in range(n - i):
            dif[j][i] = (dif[j+1][i-1] - dif[j][i-1]) / (X[j+i] - X[j])
    return dif

def evaluar_polinomio_newton(x, X, diferencias):
    n = len(X)
    resultado = diferencias[0][0]
    prod = 1.0
    for i in range(1, n):
        prod *= (x - X[i-1])
        resultado += diferencias[0][i] * prod
    return resultado

def verificar_resultados(X, Y, diferencias):
    print("\nVerificación de los resultados:")
    print(f"{'Xi':>10} | {'Yi (Real)':>10} | {'P(Xi) (Interpolado)':>20} | {'Error':>10}")
    print("-" * 60)

    for i in range(len(X)):
        valor_interpolado = evaluar_polinomio_newton(X[i], X, diferencias)
        error = abs(valor_interpolado - Y[i])
        print(f"{X[i]:>10.4f} | {Y[i]:>10.4f} | {valor_interpolado:>20.4f} | {error:>10.4e}")

def obtener_polinomio_simplificado(X, diferencias):
    n = len(X)
    coeficientes = [diferencias[0][0]]
    for i in range(1, n):
        coeficientes.append(diferencias[0][i])

    # Construir el polinomio en su forma expandida
    simbolico = [coeficientes[0]]
    for i in range(1, n):
        term = [1]
        for j in range(i):
            term = np.convolve(term, [1, -X[j]])
        term = [c * coeficientes[i] for c in term]
        simbolico = np.pad(simbolico, (len(term)-len(simbolico), 0), 'constant')
        simbolico += term

    # Mostrar el polinomio simplificado
    print("\nEl polinomio interpolador en su forma mínima es:")
    mostrar_polinomio_newton(simbolico[::-1])  # Invertir para mostrar de mayor a menor grado
    return simbolico

def mostrar_polinomio_newton(coeficientes):
    n = len(coeficientes)
    terms = []
    for i in range(n):
        coef = coeficientes[i]
        grado = n - i - 1
        if abs(coef) > 1e-8:
            if grado == 0:
                term = f"{abs(coef):.4f}"
            elif grado == 1:
                term = f"{abs(coef):.4f}x"
            else:
                term = f"{abs(coef):.4f}x^{grado}"
            if coef < 0:
                term = "- " + term
            else:
                term = "+ " + term
            terms.append(term)
    if terms:
        # Eliminar el signo '+' del primer término si es necesario
        if terms[0].startswith('+ '):
            terms[0] = terms[0][2:]
        polynomial_str = ' '.join(terms)
    else:
        polynomial_str = "0"
    print(f"f(x) = {polynomial_str}")

def interpolacion_newton():
    print("-" * 60)
    print("Interpolación de Newton")
    print("-" * 60)

    x_input = input("Ingrese los valores de x separados por espacios: ")
    X = list(map(float, x_input.strip().split()))
    y_input = input("Ingrese los valores de y separados por espacios: ")
    Y = list(map(float, y_input.strip().split()))

    if len(X) != len(Y):
        print("Los vectores x e y deben tener la misma longitud.")
        return

    # Ordenar los puntos según X
    pares_ordenados = sorted(zip(X, Y))
    X, Y = zip(*pares_ordenados)
    X = np.array(X)
    Y = np.array(Y)

    n = len(X)

    # Calcular diferencias divididas
    diferencias = calcular_diferencias_divididas(X, Y)

    # Mostrar tabla de diferencias divididas
    print("\nTabla de diferencias divididas:")
    header = f"{'Xi':>10} | {'Yi':>10}"
    for i in range(1, n):
        header += f" | {'Δ^'+str(i)+'Y':>15}"
    print(header)
    print("-" * (12 * n))

    for i in range(n):
        row = f"{X[i]:>10.4f} | {Y[i]:>10.4f}"
        for j in range(1, n - i):
            row += f" | {diferencias[i][j]:>15.4f}"
        print(row)

    # Obtener y mostrar el polinomio simplificado
    polinomio_coef = obtener_polinomio_simplificado(X, diferencias)

    # Verificar los resultados interpolados
    verificar_resultados(X, Y, diferencias)

    # Graficar el polinomio y los puntos
    x_vals = np.linspace(min(X), max(X), 500)
    y_vals = [evaluar_polinomio_newton(x, X, diferencias) for x in x_vals]

    plt.plot(x_vals, y_vals, label='Polinomio interpolador')
    plt.scatter(X, Y, color='red', label='Puntos de datos')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación de Newton con Diferencias Divididas')
    plt.legend()
    plt.grid(True)
    plt.show()

# Funciones para Interpolación de Lagrange
def poly_multiply(p1, p2):
    n1 = len(p1)
    n2 = len(p2)
    result = [0.0] * (n1 + n2 - 1)
    for i in range(n1):
        for j in range(n2):
            result[i + j] += p1[i] * p2[j]
    return result

def poly_add(p1, p2):
    n = max(len(p1), len(p2))
    result = [0.0] * n
    for i in range(n):
        coef1 = p1[i] if i < len(p1) else 0.0
        coef2 = p2[i] if i < len(p2) else 0.0
        result[i] = coef1 + coef2
    return result

def expandirLi(i, x):
    n = len(x)
    coeficientes = [1.0]  # Polinomio constante 1

    for j in range(n):
        if j != i:
            # Multiplicar coeficientes por (x - x_j)
            coeficientes = poly_multiply(coeficientes, [-x[j], 1.0])

    # Calcular el denominador
    denominador = 1.0
    for j in range(n):
        if j != i:
            denominador *= (x[i] - x[j])

    # Dividir los coeficientes por el denominador
    coeficientes = [coef / denominador for coef in coeficientes]

    return coeficientes

def reducirPolinomioLagrange(x, y):
    n = len(x)
    polinomio = [0.0]  # Inicializar polinomio de grado 0

    for i in range(n):
        coeficientesLi = expandirLi(i, x)
        # Multiplicar coeficientesLi por y[i]
        coeficientesLi = [coef * y[i] for coef in coeficientesLi]
        # Sumar al polinomio
        polinomio = poly_add(polinomio, coeficientesLi)

    return polinomio

def mostrar_polinomio_lagrange(polinomio):
    n = len(polinomio)
    terms = []
    for i in reversed(range(n)):
        coef = polinomio[i]
        if abs(coef) > 1e-6:
            # Construir el término
            if i == 0:
                term = f"{abs(coef):.4f}"
            elif i == 1:
                term = f"{abs(coef):.4f}x"
            else:
                term = f"{abs(coef):.4f}x^{i}"
            if coef < 0:
                term = "- " + term
            else:
                term = "+ " + term
            terms.append(term)
    if terms:
        # Eliminar el signo '+' del primer término si es necesario
        if terms[-1].startswith('+ '):
            terms[-1] = terms[-1][2:]
        elif terms[-1].startswith('- '):
            pass  # Mantener el signo negativo
        polynomial_str = ' '.join(reversed(terms))
    else:
        polynomial_str = "0"
    print(f"f(x) = {polynomial_str}")

def mostrarPolinomiosLi(x):
    n = len(x)
    print("Polinomios L_i(x):")
    for i in range(n):
        print(f"L_{i}(x) = ", end='')
        coeficientesLi = expandirLi(i, x)
        mostrar_polinomio_lagrange(coeficientesLi)

def evaluarPolinomio(polinomio, x_vals):
    y_vals = []
    for x in x_vals:
        y = 0.0
        for i, coef in enumerate(polinomio):
            y += coef * (x ** i)
        y_vals.append(y)
    return y_vals

def interpolacion_lagrange():
    print("-" * 60)
    print("Interpolación de Lagrange")
    print("-" * 60)
    x_input = input("Ingrese los valores de x separados por espacios: ")
    x = list(map(float, x_input.strip().split()))
    y_input = input("Ingrese los valores de y separados por espacios: ")
    y = list(map(float, y_input.strip().split()))

    if len(x) != len(y):
        print("Los vectores x e y deben tener la misma longitud.")
        return

    # Ordenar x e y basados en x
    pares_ordenados = sorted(zip(x, y))
    x, y = zip(*pares_ordenados)
    x, y = list(x), list(y)

    mostrarPolinomiosLi(x)

    polinomio = reducirPolinomioLagrange(x, y)
    print("\nPolinomio de Lagrange reducido:")
    mostrar_polinomio_lagrange(polinomio)

    # Generar valores de x para graficar
    x_min, x_max = min(x), max(x)
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = evaluarPolinomio(polinomio, x_vals)

    # Graficar
    plt.plot(x_vals, y_vals, label='Polinomio interpolante')
    plt.scatter(x, y, color='red', label='Puntos de datos')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolación de Lagrange')
    plt.legend()
    plt.grid(True)
    plt.show()

# Funciones para Interpolación con Splines Cúbicos Naturales
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

def imprimir_polinomios_reducidos_spline(splines, x_max):
    for i, s in enumerate(splines):
        a, b, c, d, x0 = s.a, s.b, s.c, s.d, s.x
        a_red = a - b*x0 + c*x0**2 - d*x0**3
        b_red = b - 2*c*x0 + 3*d*x0**2
        c_red = c - 3*d*x0
        d_red = d

        intervalo_fin = splines[i+1].x if i < len(splines) - 1 else x_max
        print(f"Polinomio {i} en el intervalo [{x0}, {intervalo_fin}]:")
        print(f"S_{i}(x) = {a_red:.4f} + {b_red:.4f} * x + {c_red:.4f} * x^2 + {d_red:.4f} * x^3\n")

def graficar_splines(splines, x_puntos, y_puntos, titulo):
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
    plt.plot(x_plot, y_plot, label=titulo)
    plt.scatter(x_puntos, y_puntos, color='red', label='Puntos de datos')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titulo)
    plt.grid(True)
    plt.show()

def interpolacion_splines_naturales():
    print("-"*60)
    print("Interpolación con Splines Cúbicos Naturales")
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
    imprimir_polinomios_reducidos_spline(splines, x_max)

    # Graficar los splines sin solicitar un valor de x
    graficar_splines(splines, x, y, 'Interpolación con Splines Cúbicos Naturales')

# Funciones para Interpolación con Splines Cúbicos Condicionados
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

def imprimir_polinomios_reducidos_condicionados(splines, x):
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

def interpolacion_splines_condicionados():
    print("-"*60)
    print("Interpolación con Splines Cúbicos Condicionados")
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
    imprimir_polinomios_reducidos_condicionados(splines, x)

    # Graficar los splines sin solicitar un valor de x
    graficar_splines(splines, x, y, 'Interpolación con Splines Cúbicos Condicionados')

# Función principal con menú
def main():
    while True:
        print("-" * 60)
        print("Interpolación de polinomios")
        print("-" * 60)
        print("Seleccione el método de interpolación:")
        print("1. Método de Newton")
        print("2. Método de Lagrange")
        print("3. Splines Cúbicos Naturales")
        print("4. Splines Cúbicos Condicionados")
        print("5. Salir")
        opcion = input("Ingrese el número de la opción deseada: ")

        if opcion == '1':
            interpolacion_newton()
        elif opcion == '2':
            interpolacion_lagrange()
        elif opcion == '3':
            interpolacion_splines_naturales()
        elif opcion == '4':
            interpolacion_splines_condicionados()
        elif opcion == '5':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, seleccione un número entre 1 y 5.")

if __name__ == "__main__":
    main()
