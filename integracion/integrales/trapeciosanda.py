# Integración: Regla de los trapecios
import numpy as np
import matplotlib.pyplot as plt


def integracion_por_funcion():
    """Realiza la integración utilizando una función ingresada por el usuario."""
    # Entrada del usuario para la función, el intervalo y los tramos
    funcion_str = input("Ingrese la función en términos de x (por ejemplo, '1 - x - 4 * x**3 + 3 * x**5'): ")
    a = float(input("Ingrese el límite inferior de integración (a): "))
    b = float(input("Ingrese el límite superior de integración (b): "))
    tramos = int(input("Ingrese el número de tramos: "))

    # Definición de la función a partir de la entrada del usuario
    fx = lambda x: eval(funcion_str, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
                                      "exp": np.exp, "log": np.log, "sqrt": np.sqrt})

    # Calculando h y valores iniciales
    h = (b - a) / tramos
    xi = a
    suma = fx(xi)

    # Imprimir encabezado
    print(f'{"Paso":<5} {"xi":<10} {"f(xi)":<15} {"Suma parcial":<15}')
    print(f'{0:<5} {xi:<10.4f} {fx(xi):<15.8f} {suma:<15.8f}')

    # Ciclo de integración por trapecios
    for i in range(1, tramos):
        xi += h
        suma += 2 * fx(xi)
        # Imprimir valores en cada paso
        print(f'{i:<5} {xi:<10.4f} {fx(xi):<15.8f} {suma:<15.8f}')

    suma += fx(b)
    area = h * (suma / 2)

    # Imprimir el último paso
    print(f'{tramos:<5} {b:<10.4f} {fx(b):<15.8f} {suma:<15.8f}')
    print('Tramos:', tramos)
    print('Integral:', area)

    # Gráfica
    graficar_funcion(fx, a, b, tramos)


def integracion_por_tabla():
    """Realiza la integración utilizando una tabla de valores (x, y) ingresada por el usuario."""
    # Entrada del usuario para los valores de x y y
    xi = list(map(float, input("Ingrese los valores de x separados por comas: ").split(',')))
    fi = list(map(float, input("Ingrese los valores de y separados por comas: ").split(',')))

    # Verificación de que ambas listas tienen el mismo tamaño
    if len(xi) != len(fi):
        print("Error: Las listas de x y y deben tener la misma longitud.")
        return

    # Imprimir encabezado
    print(f'{"Paso":<5} {"xi":<10} {"f(xi)":<15} {"Suma parcial":<15}')

    # Integración usando la regla de los trapecios para valores discretos
    suma = 0
    for i in range(len(xi) - 1):
        dx = xi[i + 1] - xi[i]
        trapecio = dx * (fi[i + 1] + fi[i]) / 2
        suma += trapecio
        # Imprimir valores en cada paso
        print(f'{i + 1:<5} {xi[i]:<10.4f} {fi[i]:<15.8f} {suma:<15.8f}')

    print('Tramos:', len(xi) - 1)
    print('Integral:', suma)

    # Gráfica
    graficar_tabla(xi, fi)


def graficar_funcion(fx, a, b, tramos):
    """Genera la gráfica de la función y el área bajo la curva."""
    muestras = tramos + 1
    xi = np.linspace(a, b, muestras)
    fi = fx(xi)

    # Línea suave
    muestraslinea = tramos * 10 + 1
    xk = np.linspace(a, b, muestraslinea)
    fk = fx(xk)

    # Graficando
    plt.plot(xk, fk, label='f(x)')
    plt.plot(xi, fi, marker='o', color='orange', label='muestras')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Integral: Regla de Trapecios')
    plt.legend()
    plt.fill_between(xi, 0, fi, color='g')
    for i in range(0, muestras, 1):
        plt.axvline(xi[i], color='w')
    plt.show()


def graficar_tabla(xi, fi):
    """Genera la gráfica de los puntos discretos y el área bajo la curva."""
    plt.plot(xi, fi, marker='o', color='orange', label='muestras')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Integral: Regla de Trapecios (Tabla de valores)')
    plt.legend()
    plt.fill_between(xi, 0, fi, color='g')
    for i in range(len(xi)):
        plt.axvline(xi[i], color='w')
    plt.show()


# Menú de selección
print("Seleccione el método de integración:")
print("1. Ingresar una función en términos de x")
print("2. Ingresar una tabla de valores de x e y")

opcion = input("Opción (1 o 2): ")

if opcion == "1":
    integracion_por_funcion()
elif opcion == "2":
    integracion_por_tabla()
else:
    print("Opción no válida.")
