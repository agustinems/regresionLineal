import numpy as np
import matplotlib.pyplot as plt


def integracion_trapecios_por_funcion():
    """Realiza la integración por el método de los trapecios usando una función ingresada por el usuario."""
    funcion_str = input("Ingrese la función en términos de x (por ejemplo, '1 - x - 4 * x**3 + 3 * x**5'): ")
    a = float(input("Ingrese el límite inferior de integración (a): "))
    b = float(input("Ingrese el límite superior de integración (b): "))
    tramos = int(input("Ingrese el número de tramos: "))

    fx = lambda x: eval(funcion_str, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
                                      "exp": np.exp, "log": np.log, "sqrt": np.sqrt})

    h = (b - a) / tramos
    xi = a
    suma = fx(xi)
    print(f'{"Paso":<5} {"xi":<10} {"f(xi)":<15} {"Suma parcial":<15}')
    print(f'{0:<5} {xi:<10.4f} {fx(xi):<15.8f} {suma:<15.8f}')

    for i in range(1, tramos):
        xi += h
        suma += 2 * fx(xi)
        print(f'{i:<5} {xi:<10.4f} {fx(xi):<15.8f} {suma:<15.8f}')

    suma += fx(b)
    area = h * (suma / 2)
    print(f'{tramos:<5} {b:<10.4f} {fx(b):<15.8f} {suma:<15.8f}')
    print('Tramos:', tramos)
    print('Integral:', area)
    graficar_funcion(fx, a, b, tramos)


def integracion_trapecios_por_tabla():
    """Realiza la integración por el método de los trapecios usando una tabla de valores."""
    xi = list(map(float, input("Ingrese los valores de x separados por comas: ").split(',')))
    fi = list(map(float, input("Ingrese los valores de y separados por comas: ").split(',')))

    if len(xi) != len(fi):
        print("Error: Las listas de x y y deben tener la misma longitud.")
        return

    print(f'{"Paso":<5} {"xi":<10} {"f(xi)":<15} {"Suma parcial":<15}')
    suma = 0
    for i in range(len(xi) - 1):
        dx = xi[i + 1] - xi[i]
        trapecio = dx * (fi[i + 1] + fi[i]) / 2
        suma += trapecio
        print(f'{i + 1:<5} {xi[i]:<10.4f} {fi[i]:<15.8f} {suma:<15.8f}')

    print('Tramos:', len(xi) - 1)
    print('Integral:', suma)
    graficar_tabla(xi, fi)


def integracion_simpson_por_funcion():
    """Realiza la integración por el método de Simpson usando una función ingresada por el usuario."""
    funcion_str = input("Ingrese la función en términos de x (por ejemplo, '1 - x - 4 * x**3 + 3 * x**5'): ")
    a = float(input("Ingrese el límite inferior de integración (a): "))
    b = float(input("Ingrese el límite superior de integración (b): "))
    tramos = int(input("Ingrese el número de tramos (debe ser par): "))

    if tramos % 2 != 0:
        print("Error: El número de tramos debe ser par para el método de Simpson.")
        return

    fx = lambda x: eval(funcion_str, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
                                      "exp": np.exp, "log": np.log, "sqrt": np.sqrt})

    h = (b - a) / tramos
    suma = fx(a) + fx(b)
    print(f'{"Paso":<5} {"xi":<10} {"f(xi)":<15} {"Suma parcial":<15}')
    print(f'{0:<5} {a:<10.4f} {fx(a):<15.8f} {suma:<15.8f}')

    for i in range(1, tramos):
        xi = a + i * h
        factor = 4 if i % 2 != 0 else 2
        suma += factor * fx(xi)
        print(f'{i:<5} {xi:<10.4f} {fx(xi):<15.8f} {suma:<15.8f}')

    area = h * (suma / 3)
    print(f'{tramos:<5} {b:<10.4f} {fx(b):<15.8f} {suma:<15.8f}')
    print('Tramos:', tramos)
    print('Integral:', area)
    graficar_funcion(fx, a, b, tramos)


def integracion_simpson_por_tabla():
    """Realiza la integración por el método de Simpson usando una tabla de valores."""
    xi = list(map(float, input("Ingrese los valores de x separados por comas: ").split(',')))
    fi = list(map(float, input("Ingrese los valores de y separados por comas: ").split(',')))

    if len(xi) != len(fi):
        print("Error: Las listas de x y y deben tener la misma longitud.")
        return
    if (len(xi) - 1) % 2 != 0:
        print("Error: El número de tramos debe ser par para el método de Simpson.")
        return

    print(f'{"Paso":<5} {"xi":<10} {"f(xi)":<15} {"Suma parcial":<15}')
    suma = fi[0] + fi[-1]
    for i in range(1, len(xi) - 1):
        factor = 4 if i % 2 != 0 else 2
        suma += factor * fi[i]
        print(f'{i:<5} {xi[i]:<10.4f} {fi[i]:<15.8f} {suma:<15.8f}')

    h = (xi[-1] - xi[0]) / (len(xi) - 1)
    area = h * (suma / 3)
    print('Integral:', area)
    graficar_tabla(xi, fi)


def graficar_funcion(fx, a, b, tramos):
    """Genera la gráfica de la función y el área bajo la curva."""
    muestras = tramos + 1
    xi = np.linspace(a, b, muestras)
    fi = fx(xi)
    muestraslinea = tramos * 10 + 1
    xk = np.linspace(a, b, muestraslinea)
    fk = fx(xk)
    plt.plot(xk, fk, label='f(x)')
    plt.plot(xi, fi, marker='o', color='orange', label='muestras')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Integral: Método de Integración')
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
    plt.title('Integral: Método de Integración (Tabla de valores)')
    plt.legend()
    plt.fill_between(xi, 0, fi, color='g')
    for i in range(len(xi)):
        plt.axvline(xi[i], color='w')
    plt.show()


# Menú de selección de método de integración
print("Seleccione el método de integración:")
print("1. Método de los Trapecios")
print("2. Método de Simpson")

metodo = input("Método (1 o 2): ")

if metodo == "1":
    print("Seleccione el tipo de integración:")
    print("1. Ingresar una función en términos de x")
    print("2. Ingresar una tabla de valores de x e y")
    opcion = input("Opción (1 o 2): ")

    if opcion == "1":
        integracion_trapecios_por_funcion()
    elif opcion == "2":
        integracion_trapecios_por_tabla()
    else:
        print("Opción no válida.")

elif metodo == "2":
    print("Seleccione el tipo de integración:")
    print("1. Ingresar una función en términos de x")
    print("2. Ingresar una tabla de valores de x e y")
    opcion = input("Opción (1 o 2): ")

    if opcion == "1":
        integracion_simpson_por_funcion()
    elif opcion == "2":
        integracion_simpson_por_tabla()
    else:
        print("Opción no válida.")
else:
    print("Método no válido.")
