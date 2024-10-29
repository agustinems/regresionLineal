
# Integración: Regla de los trapecios
# Usando una función fx()
import numpy as np
import matplotlib.pyplot as plt

fx = lambda x: 1 - x - 4 * np.pow(x,3) + 3 * np.pow(x,5)

# intervalo de integración
a = -3
b = 5
tramos = 10

h = (b-a)/tramos
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

# Resultado final
print('Tramos: ', tramos)
print('Integral: ', area)
# GRAFICA
# Puntos de muestra
muestras = tramos + 1
xi = np.linspace(a,b,muestras)
fi = fx(xi)
# Linea suave
muestraslinea = tramos*10 + 1
xk = np.linspace(a,b,muestraslinea)
fk = fx(xk)

# Graficando
plt.plot(xk,fk, label ='f(x)')
plt.plot(xi,fi, marker='o',
         color='orange', label ='muestras')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Integral: Regla de Trapecios')
plt.legend()

# Trapecios
plt.fill_between(xi,0,fi, color='g')
for i in range(0,muestras,1):
    plt.axvline(xi[i], color='w')

plt.show()

"""
#para muestras

# Integración: Regla de los trapecios
# Usando una muestras xi,fi
import numpy as np
import matplotlib.pyplot as plt

def integratrapecio_fi(xi,fi):
    ''' sobre muestras de fi para cada xi
        integral con método de trapecio
    '''
    n = len(xi)
    suma = 0
    for i in range(0,n-1,1):
        dx = xi[i+1]-xi[i]
        untrapecio = dx*(fi[i+1]+fi[i])/2
        suma = suma + untrapecio
    return(suma)

# PROGRAMA -----------------
# INGRESO
xi = [1. , 1.5, 2. , 2.5, 3. ]
fi = [0.84147098, 1.22167687, 1.28594075,
      0.94626755, 0.24442702]

# PROCEDIMIENTO
Area = integratrapecio_fi(xi,fi)

# SALIDA
print('tramos: ',len(xi)-1)
print('Integral con trapecio: ',Area)
"""