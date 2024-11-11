import numpy as np
import matplotlib.pyplot as plt

N=1000
h=0.01

y=np.zeros(N+1)
y[0] = 1
y[2]=3-4*y[1]
y[1]=(-4)/(-6+h**2)

for i in range(2, N+1):
    y[i] = y[i-1] * (2 - h**2)-y[i-2]

plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, (N+1)*h, h), y, label="Solution y_n")
plt.title("rekurencja")
plt.xlabel("nh")
plt.ylabel("y_n")
plt.legend()
plt.grid(True)
plt.savefig('plot1.png')


