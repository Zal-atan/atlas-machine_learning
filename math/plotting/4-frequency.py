#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
plt.hist(student_grades,
         bins=6,
         color="dodgerblue",
         edgecolor="black",
         range=(40,100))
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.xticks(np.arange(0, 101, 10))
plt.show()
