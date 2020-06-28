import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="darkgrid")


# Load an example dataset with long-form data

exp3a = { 'Method': ['MSOD', 'MSOD', 'MSOD', 'MSOD', 'MSE NN', 'MSE NN', 'MSE NN', 'MSE NN'], 'Error': [0.0368 , 0.0514 , 0.03063, 0.0500, 0.0863 , 0.1035 , 0.0769, 0.0939], 'NN Width': [5, 50, 500, 5000, 5, 50, 500, 5000] }
exp3b = { 'Method': ['MSOD', 'MSOD', 'MSOD', 'MSOD', 'MSE NN', 'MSE NN', 'MSE NN', 'MSE NN'], 'Error': [0.0365, 0.0514, 0.0287, 0.02611, 0.0859, 0.0996, 0.0831, 0.0782], 'NN Width': [5, 50, 500, 5000, 5, 50, 500, 5000] }

f, axes = plt.subplots(1,2)

# Plot the responses for different events and regions
ax = sns.lineplot(x="NN Width", y="Error", hue="Method", data=exp3a, ax=axes[0])
ax.set(xlabel='Width', ylabel='Error')
ax.set(title='Cifar10:10k (ADAM), ReLU')
ax.axhline(0.0763, ls='--')
ax.text(5,0.068, "MSE NNGP")
ax.set_ylim(0.00,0.16)
plt.xscale('log')


ax = sns.lineplot(x="NN Width", y="Error", hue="Method", data=exp3b, ax=axes[1])
ax.set(xlabel='Width', ylabel='Error')
ax.set(title='Cifar10:10k (SGD), ReLU')
ax.axhline(0.0763, ls='--')
ax.text(5,0.068, "MSE NNGP")
ax.set_ylim(0.00,0.16)
plt.xscale('log')
plt.show()
