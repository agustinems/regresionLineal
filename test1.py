

# Modificación del código
modified_code = '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


# Modelos no lineales
def modelo_exponencial(x, a, b):
    return a * np.exp(b * x)


def modelo_potencial(x, a, b):
    return a * np.power(x, b)


def modelo_crecimiento(x, a, b):
    return a / (1 + b * x)


# Ajuste lineal por mínimos cuadrados
def ajuste_lineal(x, y):
    slope, intercept, r_value, _, _ = linregress(x, y)
    return slope, intercept, r_value ** 2


# Cálculo de promedios
def calcular_promedios(x, y):
    return np.mean(x), np.mean(y)


# Imprimir tabla para función lineal
def imprimir_tabla_lineal(x, y, slope, intercept):
    x_cuadrado = np.round(x ** 2, 5)
    x_por_y = np.round(x * y, 5)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_cuadrado = np.sum(x_cuadrado)
    sum_x_por_y = np.sum(x_por_y)

    x_prom, y_prom = calcular_promedios(x, y)

    print("\\nTabla - Función Lineal")
    print(f"{'X':<10} {'Y':<10} {'X^2':<10} {'X*Y':<10}")
    for i in range(len(x)):
        print(f"{x[i]:<10.5f} {y[i]:<10.5f} {x_cuadrado[i]:<10.5f} {x_por_y[i]:<10.5f}")

    print(f"\\nSumatoria X: {sum_x:.5f}")
    print(f"Sumatoria Y: {sum_y:.5f}")
    print(f"Sumatoria X^2: {sum_x_cuadrado:.5f}")
    print(f"Sumatoria X*Y: {sum_x_por_y:.5f}")
    print(f"Promedio X: {x_prom:.5f}")
    print(f"Promedio Y: {y_prom:.5f}")

    print(f"\\nPendiente: {slope:.5f}")
    print(f"Intercepto: {intercept:.5f}")
    print(f"Ecuación: Y = {slope:.5f} * X + {intercept:.5f}")

    # Graficar la función lineal
    x_smooth = np.linspace(min(x), max(x), 500)
    y_smooth = slope * x_smooth + intercept
    plt.figure()
    plt.scatter(x, y, label='Puntos originales', color='blue')
    plt.plot(x_smooth, y_smooth, label='Ajuste Lineal', color='red')
    plt.title('Ajuste Lineal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


# Gráficas
def graficar_puntos(x, y):
    plt.figure()
    plt.scatter(x, y, label='Puntos originales', color='blue')
    plt.plot(x, y, label='Línea que une los puntos', color='green', linestyle='--')  # Línea que une los puntos
    plt.title('Gráfico de Puntos')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


# Ajuste no lineal
def graficar_ajuste_no_lineal(x, y, x_smooth, y_smooth, titulo):
    plt.figure()
    plt.scatter(x, y, label='Puntos originales', color='blue')
    plt.plot(x_smooth, y_smooth, label='Curva ajustada', color='red')
    plt.title(titulo)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def ajustar_modelo_no_lineal(x, y, modelo):
    if modelo == 'exponencial':
        popt, _ = curve_fit(modelo_exponencial, x, y)
        a, b = popt
        x_smooth = np.linspace(min(x), max(x), 500)
        y_smooth = modelo_exponencial(x_smooth, a, b)
        titulo = f"Ajuste Exponencial: Y = {a:.5f} * exp({b:.5f} * X)"
        graficar_ajuste_no_lineal(x, y, x_smooth, y_smooth, titulo)
    elif modelo == 'potencial':
        popt, _ = curve_fit(modelo_potencial, x, y)
        a, b = popt
        x_smooth = np.linspace(min(x), max(x), 500)
        y_smooth = modelo_potencial(x_smooth, a, b)
        titulo = f"Ajuste Potencial: Y = {a:.5f} * X^{b:.5f}"
        graficar_ajuste_no_lineal(x, y, x_smooth, y_smooth, titulo)
    elif modelo == 'crecimiento':
        popt, _ = curve_fit(modelo_crecimiento, x, y)
        a, b = popt
        x_smooth = np.linspace(min(x), max(x), 500)
        y_smooth = modelo_crecimiento(x_smooth, a, b)
        titulo = f"Ajuste de Crecimiento: Y = {a:.5f} / (1 + {b:.5f} * X)"
        graficar_ajuste_no_lineal(x, y, x_smooth, y_smooth, titulo)


# Menú de selección
def seleccionar_modelo_no_lineal(x, y):
    while True:
        print("\\nSeleccione el modelo a utilizar:")
        print("1. Modelo Exponencial")
        print("2. Modelo Potencial")
        print("3. Modelo de Crecimiento")
        print("4. Volver a ingresar otro problema")
        opcion = input("Ingrese el número de la opción deseada: ")

        if opcion == '1':
            ajustar_modelo_no_lineal(x, y, 'exponencial')
        elif opcion == '2':
            ajustar_modelo_no_lineal(x, y, 'potencial')
        elif opcion == '3':
            ajustar_modelo_no_lineal(x, y, 'crecimiento')
        elif opcion == '4':
            return  # Regresar al menú principal
        else:
            print("Opción no válida.")


# Función principal
def main():
    while True:
        try:
            x = np.array(list(map(float, input("Ingrese valores de X separados por espacio: ").split())))
            y = np.array(list(map(float, input("Ingrese valores de Y separados por espacio: ").split())))

            if len(x) != len(y):
                print("Error: La cantidad de elementos en X y Y debe ser la misma.")
                continue

            graficar_puntos(x, y)  # Aquí se grafican los puntos con la línea conectando

            es_lineal = input("¿La función es lineal? (si/no): ").strip().lower()
            if es_lineal == 'si':
                slope, intercept, _ = ajuste_lineal(x, y)
                imprimir_tabla_lineal(x, y, slope, intercept)
            else:
                seleccionar_modelo_no_lineal(x, y)

            if input("¿Desea resolver otro problema? (si/no): ").strip().lower() != 'si':
                break
        except ValueError:
            print("Error: Por favor ingrese solo números válidos.")
        except Exception as e:
            print(f"Se produjo un error: {e}")

if __name__ == "__main__":
    main()
'''

# Guardar el código modificado en un archivo nuevo
output_file_path = '/mnt/data/test1_modified.py'
with open(output_file_path, 'w') as file:
    file.write(modified_code)

output_file_path
