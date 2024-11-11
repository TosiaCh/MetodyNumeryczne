import numpy as np
import matplotlib.pyplot as plt

N = 1000
h = 0.01
#Ay=b
def macierz(N, h):
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    b[0] = 1
    b[N] = -1
    A[0,0]=1
    for i in range(1, N):
        A[i, i-1] = 1
        A[i, i] = -2 + h**2
        A[i, i+1] = 1
 
    A[N,N-1] = 1
    A[N,N] = -2
    

    return A, b


def algorytmThomasa(A, b):
    N = len(b) - 1

    # kopiowanie diagonali
    a = np.copy(np.diag(A, k=-1))
    b_diag = np.copy(np.diag(A, k=0))
    c = np.copy(np.diag(A, k=1))
    
    # pętla przód (forward pass)
    for i in range(1, N+1):
        m = a[i-1] / b_diag[i-1]
        b_diag[i] = b_diag[i] - m * c[i-1]
        b[i] = b[i] - m * b[i-1]
    
    # pętla wsteczna (back-substitution)
    y = np.zeros(N+1)
    y[N] = b[N] / b_diag[N]
    for i in range(N-1, -1, -1):
        y[i] = (b[i] - c[i] * y[i+1]) / b_diag[i]

    return y



A, b = macierz(N, h)
wynik = algorytmThomasa(A, b)


# wykres

plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, (N+1)*h, h), wynik, label="Solution y_n")
plt.title("algorytm thomasa")
plt.xlabel("nh")
plt.ylabel("y_n")
plt.legend()
plt.grid(True)
plt.savefig('plot1.png')
