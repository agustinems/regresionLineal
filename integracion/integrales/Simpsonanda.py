import numpy as np
import matplotlib.pyplot as plt


# Definición de la función de Simpson 1/3 para una función
def simpson13_fx(fx, a, b, tramos):
    h = (b - a) / tramos
    xi = a
    area = 0
    print("Procedimiento de la integración por Simpson 1/3:")
    for i in range(0, tramos, 2):
        deltaA = (h / 3) * (fx(xi) + 4 * fx(xi + h) + fx(xi + 2 * h))
        area += deltaA
        print(f"Tramo {i // 2 + 1}: x0 = {xi:.4f}, x1 = {xi + h:.4f}, x2 = {xi + 2 * h:.4f}, "
              f"Área del tramo = {deltaA:.4f}")
        xi += 2 * h
    print(f"\nÁrea total: {area:.4f}")
    return area


# Definición de la función de Simpson 1/3 para listas de puntos
def simpson13_fi(xi, fi, tolera=1e-10):
    n = len(xi)
    area = 0
    i = 0
    print("Procedimiento de la integración con puntos dados por Simpson 1/3:")
    while i <= n - 3:
        h = xi[i + 1] - xi[i]
        dh = abs(h - (xi[i + 2] - xi[i + 1]))
        if dh < tolera:  # Tramos iguales
            unS13 = (h / 3) * (fi[i] + 4 * fi[i + 1] + fi[i + 2])
            area += unS13
            print(f"Tramo {i // 2 + 1}: x0 = {xi[i]:.4f}, x1 = {xi[i + 1]:.4f}, x2 = {xi[i + 2]:.4f}, "
                  f"Área del tramo = {unS13:.4f}")
        else:
            return f'Tramos desiguales en i: {i} , {i + 2}'
        i += 2
    if i < n - 1:  # Incompleto, faltan tramos por calcular
        return f'Tramos incompletos, faltan {(n - 1) - i} tramos'
    print(f"\nÁrea total: {area:.4f}")
    return area


# MENÚ PARA EL USUARIO
print("Seleccione el tipo de entrada:")
print("1. Ingresar una función")
print("2. Ingresar listas de valores de x y y")

opcion = int(input("Opción: "))

if opcion == 1:
    # INGRESO: función
    fx_input = input("Ingrese la función de f(x) en términos de 'x': ")
    fx = eval("lambda x: " + fx_input, {"cos": np.cos, "sin": np.sin, "tan": np.tan, "exp": np.exp, "log": np.log})

    # Intervalo de integración y cantidad de tramos
    a = float(input("Ingrese el límite inferior de integración (a): "))
    b = float(input("Ingrese el límite superior de integración (b): "))
    tramos = int(input("Ingrese el número de tramos (debe ser par): "))

    # Validar cantidad de tramos pares
    while tramos % 2 != 0:
        print("El número de tramos debe ser par.")
        tramos = int(input("Ingrese un número de tramos par: "))

    # Cálculo del área con Simpson 1/3
    area = simpson13_fx(fx, a, b, tramos)

    # Mejorar la gráfica
    muestras = tramos + 1
    xi = np.linspace(a, b, muestras)
    fi = fx(xi)

    # Graficar la función con áreas y puntos
    xk = np.linspace(a, b, 1000)
    fk = fx(xk)
    plt.plot(xk, fk, label="f(x)", color="blue", linewidth=1.5)
    plt.fill_between(xk, fk, color="lightyellow", alpha=0.3)

    # Graficar los tramos con relleno alternado
    for i in range(0, muestras - 1, 2):
        x_tramo = np.linspace(xi[i], xi[i + 2], 100)
        f_tramo = fx(x_tramo)
        color = 'lightgreen' if (i // 2) % 2 == 0 else 'lightblue'
        plt.fill_between(x_tramo, f_tramo, color=color, alpha=0.5)

    # Marcar los puntos de integración
    plt.plot(xi, fi, 'ro', label="Puntos de integración")
    for i in range(muestras):
        plt.text(xi[i], fi[i], f"({xi[i]:.2f}, {fi[i]:.2f})", ha='center', fontsize=8)

    # Personalización de la gráfica
    plt.title("Integral de f(x) usando la Regla de Simpson 1/3")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

elif opcion == 2:
    # INGRESO: listas de valores de x y y
    xi = list(map(float, input("Ingrese los valores de x separados por comas: ").split(',')))
    fi = list(map(float, input("Ingrese los valores de f(x) correspondientes a x, separados por comas: ").split(',')))

    # Verificar que las listas tengan la misma longitud y longitud impar
    if len(xi) != len(fi):
        print("Error: Las listas de x y f(x) deben tener la misma longitud.")
    elif (len(xi) - 1) % 2 != 0:
        print("Error: El número de tramos (len(x)-1) debe ser par.")
    else:
        # Cálculo del área con Simpson 1/3
        area = simpson13_fi(xi, fi)

        # Mostrar área y gráfica
        print("Integral con Simpson 1/3: ", area)

        # Gráfica de los puntos
        plt.plot(xi, fi, 'ro-', label="Puntos (x, f(x))")
        plt.fill_between(xi, fi, color="lightgreen", alpha=0.3)
        plt.title("Integral con Simpson 1/3 (Listas de Puntos)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.legend()
        plt.show()

else:
    print("Opción no válida.")
