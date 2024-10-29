import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Definir la ecuación diferencial a resolver. El usuario puede personalizar esta función.
def f(x, y):
    return eval(equation)


# Método de Runge-Kutta de segundo orden con tabla de resultados detallada
def runge_kutta_2nd_order_table(x0, y0, h, n):
    x, y = x0, y0
    results = {
        "x_i": [x],
        "y_i": [y],
        "K1": [None],
        "K2": [None]
    }

    for i in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h, y + k1)
        y += 0.5 * (k1 + k2)
        x += h
        results["x_i"].append(x)
        results["y_i"].append(y)
        results["K1"].append(k1)
        results["K2"].append(k2)

    return results


# Función para resolver el método de segundo orden y mostrar la tabla
def runge_kutta_2nd_order_solver(x0, y0, h, x_end):
    n = int((x_end - x0) / h)
    return runge_kutta_2nd_order_table(x0, y0, h, n)


# Ingreso de datos por consola
print("Método de Runge-Kutta de segundo orden con tabla de resultados")
equation = input("Ingrese la ecuación diferencial dy/dx en términos de x y y (ejemplo: 'np.exp(0.8 * x) - 0.5 * y'): ")
x0 = float(input("Ingrese el valor inicial de x (x0): "))
y0 = float(input("Ingrese el valor inicial de y (y0): "))
h = float(input("Ingrese el tamaño del paso (h): "))
x_end = float(input("Ingrese el valor final de x: "))

# Resolver usando el método de segundo orden y obtener la tabla de resultados
results = runge_kutta_2nd_order_solver(x0, y0, h, x_end)

# Mostrar la tabla de resultados
df = pd.DataFrame(results)
print("\nTabla de resultados:")
print(df)

# Imprimir la ecuación diferencial analítica aproximada
print("\nEcuación diferencial analítica aproximada encontrada:")
print(
    "Esta es una aproximación numérica; para una ecuación analítica exacta, deberías integrar la función manualmente o usar un CAS.")

# Graficar la solución aproximada
x_vals = df["x_i"]
y_vals = df["y_i"]
plt.plot(x_vals, y_vals, marker='o', label="Runge-Kutta de 2° orden")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solución aproximada con el método de Runge-Kutta de 2° orden")
plt.legend()
plt.grid()
plt.show()
