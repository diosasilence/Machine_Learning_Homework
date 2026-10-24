import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,2,50)
y1 = np.log(1+x)
y2=x
y3 = np.exp(x)-1

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)

plt.xlim(-1, 2)
plt.ylim(-1, 2)

plt.grid(True, linestyle='-', alpha=0.6)

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)



plt.show()
