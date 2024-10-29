import numpy as np
import matplotlib.pyplot as plt


def euler(f, t0, y0, t_final, h):
    """
    Método de Euler para resolver la ecuación diferencial y' = f(t, y)

    Parámetros:
    f       -- función que representa la ecuación diferencial y' = f(t, y)
    t0      -- valor inicial de t
    y0      -- valor inicial de y
    t_final -- valor final de t
    h       -- tamaño del paso

    Retorna:
    t_values -- array con los valores de t
    y_values -- array con los valores de y calculados en los puntos t
    """

    # Inicializar las listas para t y y
    t_values = np.arange(t0, t_final + h, h)
    y_values = np.zeros(len(t_values))

    # Condiciones iniciales
    y_values[0] = y0

    # Imprimir encabezado de la tabla
    print(f"{'Iteración':<10} {'x':<10} {'y (aproximado)':<20}")
    print("-" * 40)

    # Imprimir valores iniciales
    print(f"{0:<10} {t0:<10.4f} {y0:<20.6f}")

    # Aplicar el método de Euler
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]
        y_values[i] = y + h * f(t, y)

        # Imprimir los valores de cada iteración
        print(f"{i:<10} {t_values[i]:<10.4f} {y_values[i]:<20.6f}")

    return t_values, y_values


# Pedir al usuario la ecuación diferencial
funcion_str = input("Ingrese la función en términos de t y y (ejemplo: np.exp(0.8 * t) - 0.5 * y): ")
f = eval("lambda t, y: " + funcion_str)

# Pedir al usuario los parámetros iniciales
t0 = float(input("Ingrese el valor inicial de t: "))
y0 = float(input("Ingrese el valor inicial de y: "))
t_final = float(input("Ingrese el valor final de t: "))
h = float(input("Ingrese el tamaño del paso h: "))

# Aplicar el método de Euler
t_values, y_values = euler(f, t0, y0, t_final, h)

# Graficar los resultados
plt.plot(t_values, y_values, label="Euler Method", color="blue")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title(f"Método de Euler para y' = {funcion_str}")
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()
