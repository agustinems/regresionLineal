# Lagrange


import matplotlib.pyplot as plt
import numpy as np

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

def mostrarPolinomio(polinomio):
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
        mostrarPolinomio(coeficientesLi)

def evaluarPolinomio(polinomio, x_vals):
    y_vals = []
    for x in x_vals:
        y = 0.0
        for i, coef in enumerate(polinomio):
            y += coef * (x ** i)
        y_vals.append(y)
    return y_vals

def main():
    print("-" * 60)
    print("Polinomios de Lagrange")
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
    mostrarPolinomio(polinomio)

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

if __name__ == "__main__":
    main()
