#!/usr/bin/env python3
"""
Plots x -> y as a line graph, with the y axis in logarithmic scale.
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y)
plt.yscale('log')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')
plt.axis(xmin=0, xmax=28650)
plt.show()
