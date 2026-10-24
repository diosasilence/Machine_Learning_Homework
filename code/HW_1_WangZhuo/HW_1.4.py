import numpy as np
import matplotlib.pyplot as plt

def gauss_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

mu1, sigma1 = 0, 1
mu2, sigma2 = 2, 1.5


x_min = min(mu1 - 4 * sigma1, mu2 - 4 * sigma2)
x_max = max(mu1 + 4 * sigma1, mu2 + 4 * sigma2)
x = np.linspace(x_min, x_max, 1000)


y1 = gauss_pdf(x, mu1, sigma1)
y2 = gauss_pdf(x, mu2, sigma2)

plt.figure(figsize=(8, 8))
plt.plot(x, y1)
plt.plot(x, y2)

plt.show()