import numpy as np
import matplotlib.pyplot as plt

def macierz(N,h):
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    b[0]=1
    b[N]=0

    for i in range(1, N):
        A[i, i-1] = 1/h**2
        A[i, i] = -2/h**2 + 1
        A[i, i+1] = 1/h**2
 
    A[0, 0] = 1
    A[N, N] = 1



    return A, b


def algorytmThomasa(A, b):

    N = len(b)
    
    a = np.copy(np.diag(A, k=-1))
    c = np.copy(np.diag(A, k=1))
    d = np.copy(np.diag(A))

    for i in range(1, N):
        m = a[i-1] / d[i-1]
        d[i] -= m * c[i-1]
        b[i] -= m * b[i-1]

    x = np.zeros(N)
    x[-1] = b[-1] / d[-1]

    for i in range(N-2, -1, -1):
        x[i] = (b[i] - c[i] * x[i+1]) / d[i]

    return x



N = 1000
h = 0.01

A, b = macierz(N, h)
wynik = algorytmThomasa(A, b)


#wykres
nhWartosci = np.arange(0, N+1) * h
plt.plot(nhWartosci, wynik)
plt.title('Rozwiązanie równań różniczkowych (metoda Thomasa)')
plt.xlabel('n * h')
plt.ylabel('y_n')
plt.grid(True)
plt.savefig('plot1.png')