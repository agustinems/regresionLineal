import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit


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

    print("\nTabla - Función Lineal")
    print(f"{'X':<10} {'Y':<10} {'X^2':<10} {'X*Y':<10}")
    for i in range(len(x)):
        print(f"{x[i]:<10.5f} {y[i]:<10.5f} {x_cuadrado[i]:<10.5f} {x_por_y[i]:<10.5f}")

    print(f"\nSumatoria X: {sum_x:.5f}")
    print(f"Sumatoria Y: {sum_y:.5f}")
    print(f"Sumatoria X^2: {sum_x_cuadrado:.5f}")
    print(f"Sumatoria X*Y: {sum_x_por_y:.5f}")
    print(f"Promedio X: {x_prom:.5f}")
    print(f"Promedio Y: {y_prom:.5f}")

    print(f"\nPendiente (a1): {slope:.5f}")
    print(f"Intercepto (a0): {intercept:.5f}")
    print(f"Ecuación ajustada: Y = {slope:.5f} * X + {intercept:.5f}")

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
    plt.plot(x, y, label='Línea que une los puntos', color='green', linestyle='--')
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


def imprimir_tabla_exponencial(x, y):
    # Calculando ln(Y)
    ln_y = np.round(np.log(y), 5)

    # Calculando X² y X*ln(Y)
    x_cuadrado = np.round(x ** 2, 5)
    x_por_ln_y = np.round(x * ln_y, 5)

    # Sumatorias
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_ln_y = np.sum(ln_y)
    sum_x_cuadrado = np.sum(x_cuadrado)
    sum_x_por_ln_y = np.sum(x_por_ln_y)

    # Promedios
    x_prom = np.mean(x)
    ln_y_prom = np.mean(ln_y)

    # Regresión lineal para ln(Y) vs X
    slope, intercept, _, _, _ = linregress(x, ln_y)
    a = np.exp(intercept)
    b = slope

    # Imprimir la tabla
    print("\nTabla - Modelo Exponencial")
    print(f"{'Xᵢ':<10} {'Yᵢ':<10} {'Ln Yᵢ':<10} {'Xᵢ²':<10} {'Xᵢ*Ln Yᵢ':<15}")
    for i in range(len(x)):
        print(f"{x[i]:<10.5f} {y[i]:<10.5f} {ln_y[i]:<10.5f} {x_cuadrado[i]:<10.5f} {x_por_ln_y[i]:<15.5f}")

    # Imprimir sumatorias y promedios
    print(
        f"\nSumatorias: ΣX = {sum_x:.5f}, ΣY = {sum_y:.5f}, ΣLn Y = {sum_ln_y:.5f}, ΣX² = {sum_x_cuadrado:.5f}, ΣX*Ln Y = {sum_x_por_ln_y:.5f}")
    print(f"Promedios: X̄ = {x_prom:.5f}, (Ln Ȳ) = {ln_y_prom:.5f}")
    print(f"\nParámetros del modelo exponencial:")
    print(f"a = e^({intercept:.5f}) = {a:.5f}")
    print(f"b = {b:.5f}")
    print(f"Ecuación ajustada: Y = {a:.5f} * exp({b:.5f} * X)")

    # Graficar el ajuste exponencial
    x_smooth = np.linspace(min(x), max(x), 500)
    y_smooth = a * np.exp(b * x_smooth)
    titulo = f"Ajuste Exponencial: Y = {a:.5f} * exp({b:.5f} * X)"
    graficar_ajuste_no_lineal(x, y, x_smooth, y_smooth, titulo)


def imprimir_tabla_potencial(x, y):
    # Calculando Log(X) y Log(Y)
    log_x = np.round(np.log10(x), 5)
    log_y = np.round(np.log10(y), 5)

    # Calculando (Log X)² y (Log X)*(Log Y)
    x_cuadrado = np.round(log_x ** 2, 5)
    x_por_y = np.round(log_x * log_y, 5)

    # Sumatorias
    sum_log_x = np.sum(log_x)
    sum_log_y = np.sum(log_y)
    sum_x_cuadrado = np.sum(x_cuadrado)
    sum_x_por_y = np.sum(x_por_y)

    # Promedios
    log_x_prom = np.mean(log_x)
    log_y_prom = np.mean(log_y)

    # Regresión lineal para Log(Y) vs Log(X)
    slope, intercept, _, _, _ = linregress(log_x, log_y)
    a = 10 ** intercept
    b = slope

    # Imprimir la tabla
    print("\nTabla - Modelo Potencial")
    print(f"{'Xᵢ':<10} {'Log Xᵢ':<10} {'Yᵢ':<10} {'Log Yᵢ':<10} {'(Log Xᵢ)²':<12} {'Log Xᵢ * Log Yᵢ':<18}")
    for i in range(len(x)):
        print(
            f"{x[i]:<10.5f} {log_x[i]:<10.5f} {y[i]:<10.5f} {log_y[i]:<10.5f} {x_cuadrado[i]:<12.5f} {x_por_y[i]:<18.5f}")

    # Imprimir sumatorias y promedios
    print(
        f"\nSumatorias: ΣLog X = {sum_log_x:.5f}, ΣLog Y = {sum_log_y:.5f}, Σ(Log X)² = {sum_x_cuadrado:.5f}, ΣLog X*Log Y = {sum_x_por_y:.5f}")
    print(f"Promedios: (Log X̄) = {log_x_prom:.5f}, (Log Ȳ) = {log_y_prom:.5f}")
    print(f"\nParámetros del modelo potencial:")
    print(f"a = 10^({intercept:.5f}) = {a:.5f}")
    print(f"b = {b:.5f}")
    print(f"Ecuación ajustada: Y = {a:.5f} * X^{b:.5f}")

    # Graficar el ajuste potencial
    x_smooth = np.linspace(min(x), max(x), 500)
    y_smooth = a * np.power(x_smooth, b)
    titulo = f"Ajuste Potencial: Y = {a:.5f} * X^{b:.5f}"
    graficar_ajuste_no_lineal(x, y, x_smooth, y_smooth, titulo)


def imprimir_tabla_crecimiento(x, y, a, b):
    # Calculando 1/Y
    inv_y = np.round(1 / y, 5)

    # Calculando X² y X*(1/Y)
    x_cuadrado = np.round(x ** 2, 5)
    x_por_inv_y = np.round(x * inv_y, 5)

    # Sumatorias
    sum_x = np.sum(x)
    sum_inv_y = np.sum(inv_y)
    sum_x_cuadrado = np.sum(x_cuadrado)
    sum_x_por_inv_y = np.sum(x_por_inv_y)

    # Promedios
    x_prom = np.mean(x)
    inv_y_prom = np.mean(inv_y)

    # Regresión lineal para 1/Y vs X
    slope, intercept, _, _, _ = linregress(x, inv_y)

    # Imprimir la tabla
    print("\nTabla - Modelo de Crecimiento")
    print(f"{'Xᵢ':<10} {'Yᵢ':<10} {'1/Yᵢ':<10} {'Xᵢ²':<10} {'Xᵢ*(1/Yᵢ)':<15}")
    for i in range(len(x)):
        print(f"{x[i]:<10.5f} {y[i]:<10.5f} {inv_y[i]:<10.5f} {x_cuadrado[i]:<10.5f} {x_por_inv_y[i]:<15.5f}")

    # Imprimir sumatorias y promedios
    print(
        f"\nSumatorias: ΣX = {sum_x:.5f}, Σ(1/Y) = {sum_inv_y:.5f}, ΣX² = {sum_x_cuadrado:.5f}, ΣX*(1/Y) = {sum_x_por_inv_y:.5f}")
    print(f"Promedios: X̄ = {x_prom:.5f}, (1/Ȳ) = {inv_y_prom:.5f}")
    print(f"\nParámetros del modelo de crecimiento obtenidos por regresión lineal:")
    print(f"Pendiente (m) = {slope:.5f}")
    print(f"Intercepto (c) = {intercept:.5f}")

    # Mostrar parámetros ajustados obtenidos por curve_fit
    print(f"\nParámetros ajustados del modelo de crecimiento (usando curve_fit):")
    print(f"a = {a:.5f}")
    print(f"b = {b:.5f}")
    print(f"Ecuación ajustada: Y = {a:.5f} / (1 + {b:.5f} * X)")

    # Graficar el ajuste de crecimiento
    x_smooth = np.linspace(min(x), max(x), 500)
    y_smooth = a / (1 + b * x_smooth)
    titulo = f"Ajuste de Crecimiento: Y = {a:.5f} / (1 + {b:.5f} * X)"
    graficar_ajuste_no_lineal(x, y, x_smooth, y_smooth, titulo)


def ajustar_modelo_no_lineal(x, y, modelo):
    if modelo == 'exponencial':
        imprimir_tabla_exponencial(x, y)
    elif modelo == 'potencial':
        imprimir_tabla_potencial(x, y)
    elif modelo == 'crecimiento':
        # Ajuste usando curve_fit para mayor precisión
        popt, _ = curve_fit(modelo_crecimiento, x, y, maxfev=10000)
        a, b = popt
        imprimir_tabla_crecimiento(x, y, a, b)


# Menú de selección
def seleccionar_modelo_no_lineal(x, y):
    while True:
        print("\nSeleccione el modelo a utilizar:")
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

            graficar_puntos(x, y)  # Gráfica de los puntos

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
