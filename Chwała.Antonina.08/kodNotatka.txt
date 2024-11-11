import numpy as np
import scipy.linalg

def power_iteration(matrix, num_values=3, tolerance=1e-8, max_iterations=1000):
    n = matrix.shape[0]
    eigenvalues = []
    eigenvectors = []

    for _ in range(num_values):
        x = np.random.rand(n)
        x = x / np.linalg.norm(x, 2)

        for _ in range(max_iterations):
            y = np.dot(matrix, x)
            eigenvalue = np.dot(x, y)
            x = y / np.linalg.norm(y, 2)

            if np.linalg.norm(y - eigenvalue * x, 2) < tolerance:
                break

        eigenvalues.append(eigenvalue)
        eigenvectors.append(x)
        matrix = matrix - eigenvalue * np.outer(x, x)

    return eigenvalues, eigenvectors

def rayleigh_quotient_iteration(matrix, num_eigenvalues=3, max_iterations=100, tolerance=1e-8, precision=8):
    n = len(matrix)
    eigenvalues = []

    for _ in range(num_eigenvalues):
        b = np.random.rand(n)
        b = b / np.linalg.norm(b)
        prev_eigenvalue = 0

        for _ in range(max_iterations):
            Ab = np.dot(matrix, b)
            eigenvalue = np.dot(b, Ab) / np.dot(b, b)

            if np.abs(eigenvalue - prev_eigenvalue) < tolerance:
                eigenvalues.append(np.round(eigenvalue, precision))  
                break

            prev_eigenvalue = eigenvalue
            b = Ab / np.linalg.norm(Ab)

        matrix = matrix - eigenvalue * np.outer(b, b) / np.dot(b, b)

    return eigenvalues

def qr_iteration_eigenvalues(matrix, max_iterations=100, tolerance=1e-6):
    n = len(matrix)
    A = np.copy(matrix)
    for _ in range(max_iterations):
        Q, R = np.linalg.qr(A)
        A_next = R @ Q
        if np.allclose(A, A_next, atol=tolerance):
            break
        A = A_next
    
    eigenvalues = np.diag(A)
    return eigenvalues

# Przykładowa macierz 3x3
A = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, -1]])

# Kopia macierzy dla każdej metody
A_copy_for_power = np.copy(A)
A_copy_for_rayleigh = np.copy(A)
A_copy_for_qr = np.copy(A)

# Znajdowanie wartości własnych i wektorów własnych metodą potęgową
eigenvalues_power, eigenvectors_power = power_iteration(A_copy_for_power, num_values=3)
print("Wartości własne-metoda potęgowa:", [round(val, 8) for val in eigenvalues_power])

# Znajdowanie wartości własnych metodą Rayleigha
eigenvalues_rayleigh = rayleigh_quotient_iteration(A_copy_for_rayleigh)
print("Wartości własne-metoda Rayleighta:", eigenvalues_rayleigh)

# Znajdowanie wartości własnych metodą QR
eigenvalues_qr = qr_iteration_eigenvalues(A_copy_for_qr)
print("Wartości własne-metoda QR:", eigenvalues_qr)
