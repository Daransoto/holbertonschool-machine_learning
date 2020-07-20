#!/usr/bin/env python3
"""
Plots a histogram of student scores for a project.
"""
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, np.arange(0, 101, 10), edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.xticks(np.arange(0, 101, 10))
plt.axis([0, 100, 0, 30])
plt.show()
