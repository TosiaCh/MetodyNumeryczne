import numpy as np
import time

# Funkcje obliczające wartości własne
def iteracja_potegowa(macierz, liczba_iteracji=100):
    rozmiar = macierz.shape[0]
    wartosci_wlasne = np.empty(3)
    for indeks_wartosci in range(3):
        b_k = np.random.rand(rozmiar)
        for _ in range(liczba_iteracji):
            b_k1 = np.dot(macierz, b_k)
            b_k1_norm = np.linalg.norm(b_k1)
            b_k = b_k1 / b_k1_norm
        wartosc_wlasna = np.dot(b_k.T, np.dot(macierz, b_k))
        wartosci_wlasne[indeks_wartosci] = wartosc_wlasna
        macierz = macierz - wartosc_wlasna * np.outer(b_k, b_k)
    return wartosci_wlasne

def algorytm_qr(macierz, liczba_iteracji=100):
    rozmiar = macierz.shape[0]
    wartosci_wlasne = np.empty(3)
    for indeks_wartosci in range(3):
        Ak = np.copy(macierz)
        for _ in range(liczba_iteracji):
            Q, R = np.linalg.qr(Ak)
            Ak = R @ Q
        wartosci_wlasne[indeks_wartosci] = Ak[indeks_wartosci, indeks_wartosci]
    return wartosci_wlasne

def metoda_jacobiego(macierz, liczba_iteracji=100):
    rozmiar = macierz.shape[0]
    wartosci_wlasne = np.empty(3)
    for indeks_wartosci in range(3):
        Ak = np.copy(macierz)
        for _ in range(liczba_iteracji):
            najwieksza_poza_przekatna = np.argmax(np.abs(Ak - np.diag(np.diag(Ak))))
            i, j = np.unravel_index(najwieksza_poza_przekatna, Ak.shape)
            theta = 0.5 * np.arctan2(2 * Ak[i, j], Ak[i, i] - Ak[j, j])
            P = np.eye(rozmiar)
            P[i, i] = P[j, j] = np.cos(theta)
            P[i, j] = -np.sin(theta)
            P[j, i] = np.sin(theta)
            Ak = P.T @ Ak @ P
        wartosci_wlasne[indeks_wartosci] = Ak[indeks_wartosci, indeks_wartosci]
    return wartosci_wlasne

# Macierz z zadania
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])

# Obliczenia i pomiary czasu
start_czas = time.time()
wartosci_wlasne_potegowa = iteracja_potegowa(A, 100)
czas_potegowa = time.time() - start_czas

start_czas = time.time()
wartosci_wlasne_qr = algorytm_qr(A, 100)
czas_qr = time.time() - start_czas

start_czas = time.time()
wartosci_wlasne_jacobiego = metoda_jacobiego(A, 100)
czas_jacobiego = time.time() - start_czas

# Wypisanie wyników
print("Wartości własne (Iteracja Potęgowa):", wartosci_wlasne_potegowa)
print("Czas wykonania (Iteracja Potęgowa):", czas_potegowa, "sekund")
print("Wartości własne (Algorytm QR):", wartosci_wlasne_qr)
print("Czas wykonania (Algorytm QR):", czas_qr, "sekund")
print("Wartości własne (Metoda Jacobiego):", wartosci_wlasne_jacobiego)
print("Czas wykonania (Metoda Jacobiego):", czas_jacobiego, "sekund")
# Znalezienie i wypisanie najszybszej metody
najszybsza_metoda = min(czas_potegowa, czas_qr, czas_jacobiego)
nazwa_najszybszej_metody = ""

if najszybsza_metoda == czas_potegowa:
    nazwa_najszybszej_metody = "Iteracja Potęgowa"
elif najszybsza_metoda == czas_qr:
    nazwa_najszybszej_metody = "Algorytm QR"
elif najszybsza_metoda == czas_jacobiego:
    nazwa_najszybszej_metody = "Metoda Jacobiego"

print(f"Najszybsza metoda: {nazwa_najszybszej_metody} (czas: {najszybsza_metoda} sekund)")
nazwa_najszybszej_metody, najszybsza_metoda
