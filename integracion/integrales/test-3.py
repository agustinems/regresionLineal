# Integración: Regla Simpson 1/3
import numpy as np
import matplotlib.pyplot as plt

# INGRESO:
fx = lambda x: 1-x-4*x**3+3*x**5  # función f(x)

# intervalo de integración
a = -3
b = 5
tramos = 10

# Validar cantidad de tramos pares
esimpar = tramos % 2
while (esimpar == 1):
    print('tramos: ', tramos)
    tramos = int(input('tramos debe ser par: '))
    esimpar = tramos % 2

# PROCEDIMIENTO
# Regla de Simpson 1/3
h = (b - a) / tramos
xi = a
area = 0
for i in range(0, tramos, 2):
    deltaA = (h / 3) * (fx(xi) + 4 * fx(xi + h) + fx(xi + 2 * h))
    area = area + deltaA
    xi = xi + 2 * h

# SALIDA
print('tramos:', tramos)
print('Integral: ', area)

# GRAFICA
# fx muestras por tramo
muestras = tramos + 1
xi = np.linspace(a, b, muestras)
fi = fx(xi)
fi0 = np.zeros(muestras)  # linea base

# fx suave aumentando muestras
muestrasfxSuave = 4 * tramos + 1
xk = np.linspace(a, b, muestrasfxSuave)
fk = fx(xk)

# Relleno
for i in range(0, muestras - 1, 2):
    relleno = 'lightgreen'
    if (i / 2) % 2 == 0:  # i/2 par
        relleno = 'lightblue'
    xktramo = xk[i * 4:(i + 2) * 4 + 1]
    fktramo = fk[i * 4:(i + 2) * 4 + 1]
    plt.fill_between(xktramo, fktramo, fktramo * 0, color=relleno)

# Funcion f(x)
plt.plot(xk, fk, label='f(x)')
plt.plot(xi, fi, 'o', label='f(xi)')

# Divisiones entre Simpson 1/3
for i in range(0, muestras, 1):
    tipolinea = 'dotted'
    if i % 2 == 0:  # i par
        tipolinea = 'dashed'
    plt.vlines(xi[i], fi0[i], fi[i],
               linestyle=tipolinea)

plt.axhline(0)
plt.xlabel('x')
plt.ylabel('f')
plt.title('Integral: Regla de Simpson 1/3')
plt.legend()
plt.show()

"""
# Integración Simpson 1/3
# Usando una muestras xi,fi
import numpy as np
import matplotlib.pyplot as plt

def integrasimpson13_fi(xi,fi,tolera = 1e-10):
    ''' sobre muestras de fi para cada xi
        integral con método de Simpson 1/3
        respuesta es np.nan para tramos desiguales,
        no hay suficientes puntos.
    '''
    n = len(xi)
    i = 0
    suma = 0
    while not(i>=(n-2)):
        h = xi[i+1]-xi[i]
        dh = abs(h - (xi[i+2]-xi[i+1]))
        if dh<tolera:# tramos iguales
            unS13 = (h/3)*(fi[i]+4*fi[i+1]+fi[i+2])
            suma = suma + unS13
        else:  # tramos desiguales
            suma = 'tramos desiguales en i: '+str(i)
            suma = suma ' , '+str(i+2)
        i = i + 2
    if i<(n-1): # incompleto, faltan tramos por calcular
        suma = 'tramos incompletos, faltan '
        suma = suma ++str((n-1)-i)+' tramos'
    return(suma)

# PROGRAMA -----------------
# INGRESO
xi = [1. , 1.5, 2. , 2.5, 3.]
fi = [0.84147098, 1.22167687, 1.28594075,
      0.94626755, 0.24442702]
# PROCEDIMIENTO
Area = integrasimpson13_fi(xi,fi)

# SALIDA
print('tramos: ',len(xi)-1)
print('Integral con Simpson 1/3: ',Area)
if type(Area)==str:
    print('  Revisar errores...')
"""