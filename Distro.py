import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import norm

# This is taken from the famous study, How Big is a Burrito
# https://x.com/tanayj/status/1806423072238616781 
#
mean = 21
std_dev = 3
x = np.linspace(-2*std_dev+mean, 2*std_dev+mean, 1000)
y_custom = norm.pdf(x, loc=mean, scale=std_dev)

fig = plt.figure(figsize=(15,4))
ax1 = fig.add_subplot(121)
ax1.plot(x, y_custom, label=f'Normal Distribution (mean={mean}, std_dev={std_dev})')
ax1.xaxis.set_label('X')
ax1.yaxis.set_label('Probability Density')
ax1.grid(True)
ax1.legend()

custom_samples = np.sort(norm.rvs(loc=mean, scale=std_dev, size=75))
y = custom_samples[::-1]
print(custom_samples)

ax2 = fig.add_subplot(122)
ax2.bar(np.arange(0,75), y, color='r')
ax2.axhline(mean, color='b')
ax2.xaxis.set_ticks(np.arange(0,75,5))
ax2.set_ylim(12,28)
ax2.set_xlim(0,75)
plt.show()