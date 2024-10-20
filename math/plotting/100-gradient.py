#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# your code goes here

plt.scatter(x, y, c=z, s=30, cmap="viridis")
plt.title("Mountain Elevation")
plt.xlabel("x coordinate (m)")
plt.ylabel("y coordinate (m)")
plt.colorbar(label="elevation (m)")
plt.show()
