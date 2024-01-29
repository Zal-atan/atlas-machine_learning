#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

print(fruit)

# your code here
people = ['Farrah', 'Fred', 'Felicia']
fruit_labels = ["apples", "bananas", "oranges", "peaches"]

plt.bar(people, fruit[0], color="r", width=.5)
plt.bar(people, fruit[1], bottom=fruit[0], color="yellow", width=.5)
plt.bar(people, fruit[2], bottom=fruit[0] + fruit[1], color="#ff8000", width=.5)
plt.bar(people, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2],color="#ffe5b4", width=.5)
plt.yticks(ticks=range(0, 81, 10))
plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.legend(fruit_labels)

plt.show()
