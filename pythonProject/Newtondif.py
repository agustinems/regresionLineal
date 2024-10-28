import numpy as np
import matplotlib.pyplot as plt

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
    mostrar_polinomio(simbolico[::-1])  # Invertir para mostrar de mayor a menor grado
    return simbolico

def mostrar_polinomio(coeficientes):
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

if __name__ == "__main__":
    interpolacion_newton()
