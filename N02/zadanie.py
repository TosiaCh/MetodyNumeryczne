import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def discretization_method_a(x, h):
    return (f(x + h) - f(x)) / h

def discretization_method_b(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def discretization_method_c(x, h):
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)

def compute_error(x, h_values, method):
    errors = []
    for h in h_values:
        exact_derivative = np.cos(x) 
        numerical_derivative = method(x, h)
        error = np.abs(exact_derivative - numerical_derivative)
        errors.append(error)
    return errors

h_values = np.logspace(-16, 0, 100)
x_values = np.array([1, np.pi / 2])

errors_a = np.zeros_like(h_values)
for x in x_values:
    errors_a += compute_error(x, h_values, discretization_method_a)

errors_b = np.zeros_like(h_values)
for x in x_values:
    errors_b += compute_error(x, h_values, discretization_method_b)

errors_c = np.zeros_like(h_values)
for x in x_values:
    errors_c += compute_error(x, h_values, discretization_method_c)

optimal_h_a = h_values[np.argmin(errors_a)]
min_error_a = np.min(errors_a)
optimal_h_b = h_values[np.argmin(errors_b)]
min_error_b = np.min(errors_b)
optimal_h_c = h_values[np.argmin(errors_c)]
min_error_c = np.min(errors_c)

plt.loglog(h_values, errors_a, label='Metoda (a)')
plt.loglog(h_values, errors_b, label='Metoda (b)')
plt.loglog(h_values, errors_c, label='Metoda (c)')
plt.scatter(optimal_h_a, min_error_a, color='red')
plt.scatter(optimal_h_b, min_error_b, color='green')
plt.scatter(optimal_h_c, min_error_c, color='blue')
plt.xlabel('h')
plt.ylabel('|Dh f(x) - f\'(x)|')
plt.legend()
plt.title('Błąd w zależnośći od róznych metod dyskretyzacji')
plt.savefig('plot.png')

print("Metoda (a): Optymalne h =", optimal_h_a, "Błąd minimalny =", min_error_a)
print("Metoda (b): Optymalne h =", optimal_h_b, "Błąd minimalny =", min_error_b)
print("Metoda (c): Optymalne h =", optimal_h_c, "Błąd minimalny =", min_error_c)
