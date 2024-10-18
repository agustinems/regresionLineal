import numpy as np
import sympy as sp

def diferencia_tres_puntos_funcion(funcion, a, b, h):
    """
    Método de diferenciación numérica utilizando la fórmula de tres puntos para una función.
    """
    x_vals = np.arange(a, b + h, h)
    y_vals = funcion(x_vals)

    dy = diferencia_tres_puntos(x_vals, y_vals)
    return x_vals, dy

def diferencia_tres_puntos(x, y):
    """
    Método de diferenciación numérica utilizando la fórmula de tres puntos.
    """
    n = len(x)
    h = x[1] - x[0]  # Se asume que h es constante

    dy = np.zeros(n)

    # Fórmula de tres puntos hacia adelante para el primer punto
    dy[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * h)

    # Fórmula de tres puntos centrada para los puntos internos
    for i in range(1, n - 1):
        dy[i] = (y[i + 1] - y[i - 1]) / (2 * h)

    # Fórmula de tres puntos hacia atrás para el último punto
    dy[-1] = (y[-3] - 4 * y[-2] + 3 * y[-1]) / (2 * h)

    return dy

def diferencia_cinco_puntos_funcion(funcion, a, b, h):
    """
    Método de diferenciación numérica utilizando la fórmula de cinco puntos para una función.
    """
    x_vals = np.arange(a, b + h, h)
    y_vals = funcion(x_vals)

    dy = diferencia_cinco_puntos(x_vals, y_vals)
    return x_vals, dy

def diferencia_cinco_puntos(x, y):
    """
    Método de diferenciación numérica utilizando la fórmula de cinco puntos.
    """
    n = len(x)
    h = x[1] - x[0]

    dy = np.zeros(n)

    if n < 5:
        raise ValueError("Se requieren al menos 5 puntos para el método de cinco puntos")

    # Fórmula de cinco puntos hacia adelante para los primeros dos puntos
    dy[0] = (-25 * y[0] + 48 * y[1] - 36 * y[2] + 16 * y[3] - 3 * y[4]) / (12 * h)
    dy[1] = (-3 * y[0] - 10 * y[1] + 18 * y[2] - 6 * y[3] + y[4]) / (12 * h)

    # Fórmula de cinco puntos centrada para los puntos internos
    for i in range(2, n - 2):
        dy[i] = (y[i - 2] - 8 * y[i - 1] + 8 * y[i + 1] - y[i + 2]) / (12 * h)

    # Fórmula de cinco puntos hacia atrás para los últimos dos puntos
    dy[-2] = (3 * y[-5] - 16 * y[-4] + 36 * y[-3] - 48 * y[-2] + 25 * y[-1]) / (12 * h)
    dy[-1] = (25 * y[-1] - 48 * y[-2] + 36 * y[-3] - 16 * y[-4] + 3 * y[-5]) / (12 * h)

    return dy

# Programa principal

print("Seleccione el método de diferenciación:")
print("1. Método de tres puntos")
print("2. Método de cinco puntos")
metodo_opcion = input("Ingresa 1 o 2: ")

if metodo_opcion == "1":
    metodo_nombre = "tres puntos"
    diferencia_funcion = diferencia_tres_puntos_funcion
    diferencia_datos = diferencia_tres_puntos
elif metodo_opcion == "2":
    metodo_nombre = "cinco puntos"
    diferencia_funcion = diferencia_cinco_puntos_funcion
    diferencia_datos = diferencia_cinco_puntos
else:
    print("Opción no válida.")
    exit()

print("Seleccione una opción:")
print("1. Ingresar una función para diferenciar")
print("2. Ingresar datos de una tabla (x, y)")
opcion = input("Ingresa 1 o 2: ")

if opcion == "1":
    # Definir la función y su derivada
    x = sp.Symbol('x')

    funcion_input = input("Ingresa la función a derivar en términos de x (por ejemplo, sin(x), exp(x), etc.): ")
    try:
        # Importar las transformaciones necesarias
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
        from sympy.utilities.lambdify import lambdify

        # Definir las funciones permitidas
        allowed_funcs = {'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'exp': np.exp,
                         'sqrt': np.sqrt, 'log': np.log, 'pi': np.pi}

        # Agregar las transformaciones para la multiplicación implícita y la conversión de xor a exponentiación
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

        # Convertir la expresión a una función simbólica
        funcion_expr = parse_expr(funcion_input, transformations=transformations, local_dict={'x': x})

        derivada_expr = sp.diff(funcion_expr, x)

        # Convertir las expresiones simbólicas a funciones numéricas utilizando numpy
        funcion = lambdify(x, funcion_expr, modules=[allowed_funcs, 'numpy'])
        derivada = lambdify(x, derivada_expr, modules=[allowed_funcs, 'numpy'])

    except (sp.SympifyError, ValueError, TypeError) as e:
        print("Error al interpretar la función. Asegúrate de ingresarla correctamente.")
        print("Detalle del error:", e)
        exit()

    # Definir el intervalo y el tamaño de paso
    a = float(input("Ingresa el extremo inferior del intervalo (a): "))
    b = float(input("Ingresa el extremo superior del intervalo (b): "))
    h = float(input("Ingresa el tamaño del paso (h): "))

    # Calcular la derivada numérica
    x_vals, derivadas_aprox = diferencia_funcion(funcion, a, b, h)

    # Calcular la derivada exacta y el error
    derivadas_exactas = derivada(x_vals)
    errores = np.abs(derivadas_aprox - derivadas_exactas)

elif opcion == "2":
    print("Ingresa los valores de x separados por comas:")
    x_vals_input = input()
    print("Ingresa los valores de y separados por comas:")
    y_vals_input = input()
    try:
        x_vals = np.array([float(x.strip()) for x in x_vals_input.split(',')])
        y_vals = np.array([float(y.strip()) for y in y_vals_input.split(',')])
    except ValueError:
        print("Error al interpretar los valores de x o y. Asegúrate de ingresarlos correctamente.")
        exit()

    if len(x_vals) != len(y_vals):
        print("Las longitudes de x e y deben ser iguales.")
        exit()

    # Verificar si el paso es uniforme
    h_values = np.diff(x_vals)
    if not np.allclose(h_values, h_values[0]):
        print("Los valores de x no están uniformemente espaciados.")
        exit()
    h = h_values[0]

    try:
        derivadas_aprox = diferencia_datos(x_vals, y_vals)
    except ValueError as e:
        print("Error al calcular la derivada:", e)
        exit()

    derivadas_exactas = [None] * len(x_vals)
    errores = [None] * len(x_vals)

else:
    print("Opción no válida.")
    exit()

# Imprimir la tabla de resultados
print("\n{:<15}{:<25}{:<25}{:<25}".format("x", "Derivada Aproximada", "Derivada Exacta", "Error"))
print("-" * 90)
for xi, dyi_aprox, dyi_exacta, error in zip(x_vals, derivadas_aprox, derivadas_exactas, errores):
    xi_str = f"{xi:.5f}"
    dyi_aprox_str = f"{dyi_aprox:.10f}"
    if dyi_exacta is not None and error is not None:
        dyi_exacta_str = f"{dyi_exacta:.10f}"
        error_str = f"{error:.10f}"
    else:
        dyi_exacta_str = "N/A"
        error_str = "N/A"
    print("{:<15}{:<25}{:<25}{:<25}".format(xi_str, dyi_aprox_str, dyi_exacta_str, error_str))
