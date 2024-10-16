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


# Imprimir tabla para la función lineal
def imprimir_tabla_lineal(x, y, slope, intercept):
    x_cuadrado = np.round(x ** 2, 5)
    x_por_y = np.round(x * y, 5)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_cuadrado = np.sum(x_cuadrado)
    sum_x_por_y = np.sum(x_por_y)
    x_prom, y_prom = calcular_promedios(x, y)

    print("\nTabla - Función Lineal")
    print(f"{'X':<10} {'Y':<10} {'X^2':<10} {'X*Y':<10}")
    print("-" * 40)
    for i in range(len(x)):
        print(f"{x[i]:<10.5f} {y[i]:<10.5f} {x_cuadrado[i]:<10.5f} {x_por_y[i]:<10.5f}")

    print(f"\nSumatoria X: {sum_x:.5f}")
    print(f"Sumatoria Y: {sum_y:.5f}")
    print(f"Sumatoria X^2: {sum_x_cuadrado:.5f}")
    print(f"Sumatoria X*Y: {sum_x_por_y:.5f}")
    print(f"X_prom: {x_prom:.5f}, Y_prom: {y_prom:.5f}")
    print(f"A1 (Pendiente): {slope:.5f}")
    print(f"A0 (Intercepto): {intercept:.5f}")

    # Imprimir la ecuación de la recta
    print(f"\nEcuación de la recta: Y = {slope:.5f} * X + {intercept:.5f}")


# Gráfica de puntos originales
def graficar_puntos(x, y):
    plt.figure()
    plt.scatter(x, y, label='Puntos originales', color='blue')
    plt.title('Puntos en el eje cartesiano')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()


# Ajustar y graficar modelo lineal
def ajustar_modelo_lineal(x, y):
    slope, intercept, r2 = ajuste_lineal(x, y)
    imprimir_tabla_lineal(x, y, slope, intercept)

    # Graficar la recta ajustada
    plt.figure()
    plt.plot(x, slope * x + intercept, label='Ajuste Lineal', color='red')
    plt.scatter(x, y, label='Datos Originales', color='blue')
    plt.title(f'Ajuste Lineal por Mínimos Cuadrados (R² = {r2:.5f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


# Imprimir tabla y función del modelo exponencial
def imprimir_tabla_exponencial(x, y, slope, intercept):
    ln_y = np.log(y)
    x_cuadrado = np.round(x ** 2, 5)
    x_por_ln_y = np.round(x * ln_y, 5)
    x_prom, y_prom = calcular_promedios(x, ln_y)
    a = np.exp(intercept)

    print("\nTabla - Modelo Exponencial")
    print(f"{'X':<10} {'Y':<10} {'X^2':<10} {'ln(Y)':<15} {'X*ln(Y)':<15}")
    print("-" * 60)
    for i in range(len(x)):
        print(f"{x[i]:<10.5f} {y[i]:<10.5f} {x_cuadrado[i]:<10.5f} {ln_y[i]:<15.5f} {x_por_ln_y[i]:<15.5f}")

    print(f"\nA1 (Pendiente): {slope:.5f}")
    print(f"A0 (Intercepto): {intercept:.5f}")
    print(f"X_prom: {x_prom:.5f}, Y_prom: {y_prom:.5f}")
    print(f"Función ajustada: Y = {a:.5f} * exp({slope:.5f} * X)")


# Imprimir tabla y función del modelo potencial
def imprimir_tabla_potencial(x, y, slope, intercept):
    x_log = np.round(np.log10(x), 5)
    y_log = np.round(np.log10(y), 5)
    x_log_cuadrado = np.round(x_log ** 2, 5)
    x_log_y_log = np.round(x_log * y_log, 5)
    x_prom, y_prom = calcular_promedios(x_log, y_log)
    a = 10 ** intercept  # Corrección: A = 10^(A0)

    print("\nTabla - Modelo Potencial")
    print(f"{'X':<10} {'Y':<10} {'log10(X)':<15} {'(log10(X))^2':<15} {'log10(Y)':<15} {'log10(X)*log10(Y)':<15}")
    print("-" * 90)
    for i in range(len(x)):
        print(
            f"{x[i]:<10.5f} {y[i]:<10.5f} {x_log[i]:<15.5f} {x_log_cuadrado[i]:<15.5f} {y_log[i]:<15.5f} {x_log_y_log[i]:<15.5f}")

    print(f"\nA1 (Pendiente): {slope:.5f}")
    print(f"A0 (Intercepto): {intercept:.5f}")
    print(f"X_prom: {x_prom:.5f}, Y_prom: {y_prom:.5f}")
    print(f"Función ajustada: Y = {a:.5f} * X^{slope:.5f}")


# Imprimir resolución del modelo de crecimiento
def imprimir_resolucion_crecimiento(x, y, slope, intercept):
    x_inv = np.round(1 / x, 5)
    y_inv = np.round(1 / y, 5)
    x_prom, y_prom = calcular_promedios(x_inv, y_inv)
    a = 1 / intercept
    b = slope * a

    print("\nTabla - Modelo de Crecimiento")
    print(f"{'1/X':<10} {'1/Y':<10}")
    print("-" * 25)
    for i in range(len(x)):
        print(f"{x_inv[i]:<10.5f} {y_inv[i]:<10.5f}")

    print(f"\nA1 (Pendiente): {slope:.5f}")
    print(f"A0 (Intercepto): {intercept:.5f}")
    print(f"X_prom: {x_prom:.5f}, Y_prom: {y_prom:.5f}")
    print(f"Función ajustada: Y = {a:.5f} / (1 + {b:.5f} * X)")


# Graficar la linealización y el ajuste
def graficar_linealizacion(x, y_log, slope, intercept, titulo):
    plt.figure()
    plt.scatter(x, y_log, label='Datos linealizados', color='orange')
    plt.plot(x, slope * x + intercept, label='Ajuste lineal', color='blue')
    plt.title(f"Linealización - {titulo}")
    plt.xlabel('X')
    plt.ylabel('log(Y) / ln(Y)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Ajustar modelos no lineales
def ajustar_modelo_no_lineal(x, y, tipo_modelo):
    if tipo_modelo == 'exponencial':
        y_log = np.log(y)
        slope, intercept, _ = ajuste_lineal(x, y_log)
        imprimir_tabla_exponencial(x, y, slope, intercept)
        graficar_linealizacion(x, y_log, slope, intercept, "Exponencial")

    elif tipo_modelo == 'potencial':
        x_log = np.log10(x)
        y_log = np.log10(y)
        slope, intercept, _ = ajuste_lineal(x_log, y_log)
        imprimir_tabla_potencial(x, y, slope, intercept)
        graficar_linealizacion(x_log, y_log, slope, intercept, "Potencial")

    elif tipo_modelo == 'crecimiento':
        x_inv = 1 / x
        y_inv = 1 / y
        slope, intercept, _ = ajuste_lineal(x_inv, y_inv)
        imprimir_resolucion_crecimiento(x, y, slope, intercept)


# Menú para seleccionar modelo no lineal
def seleccionar_modelo_no_lineal(x, y):
    while True:
        print("\nSeleccione un modelo no lineal:")
        print("1. Exponencial")
        print("2. Potencial")
        print("3. Crecimiento")
        print("4. Volver al inicio")
        opcion = int(input("Ingrese opción: "))

        if opcion in [1, 2, 3]:
            tipos = ['exponencial', 'potencial', 'crecimiento']
            ajustar_modelo_no_lineal(x, y, tipos[opcion - 1])
        elif opcion == 4:
            break


# Función principal
def main():
    while True:
        try:
            x = np.array(list(map(float, input("Ingrese valores de X separados por espacio: ").split())))
            y = np.array(list(map(float, input("Ingrese valores de Y separados por espacio: ").split())))

            if len(x) != len(y):
                print("Error: La cantidad de elementos en X y Y debe ser la misma.")
                continue

            graficar_puntos(x, y)
            es_lineal = input("¿La función es lineal? (si/no): ").strip().lower()

            if es_lineal == 'si':
                ajustar_modelo_lineal(x, y)
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
