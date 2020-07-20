#!/usr/bin/env python3
"""
Plots a stacked bar graph.
"""
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

x = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
labels = ['apples', 'bananas', 'oranges', 'peaches']
bott = np.zeros(3)
for i in range(4):
    plt.bar(x, fruit[i], 0.5, bott, color=colors[i], label=labels[i])
    bott += fruit[i]
plt.legend()
plt.ytick = 10
plt.axis(ymax=80)
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.show()
