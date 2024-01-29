#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)
# your code here
plt.plot(x, y, color='blue')
plt.yscale('log')
plt.xlim(0, 28650)
plt.title("Exponential Decay of C-14")
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.show()
