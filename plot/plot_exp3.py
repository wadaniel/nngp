import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="darkgrid")


# Load an example dataset with long-form data

exp3_msod = { 'Method': ['NN-NNGP', 'NN-NNGP', 'NN-NNGP', 'NN-NNGP', 'NN-NNGP'], 'MSOD': [0.0365, 0.0514, 0.0287, 0.0275, 0.02611], 'Width': [5, 50, 500, 2000, 5000] }

# Plot the responses for different events and regions
ax = sns.lineplot(x="Width", y="MSOD", hue="Method", data=exp3_msod)
ax.set(xlabel='Width', ylabel='MSOD')
ax.set(title='Cifar10:10k, ReLU')
plt.xscale('log')
plt.show()
#plt.ylim(0.00,0.15)
#plt.xlim(0.5,7.5)
