import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="darkgrid")


# Load an example dataset with long-form data

exp2_mse = { 'Method': ['NNGP', 'NNGP', 'NNGP', 'NNGP', 'NN', 'NN', 'NN', 'NN'], 'MSE': [0.0848, 0.0743, 0.0708, 0.0711, 0.1230, 0.1230, 0.1226, 0.1215], 'Accuracy': [0.4040 , 0.4420, 0.4522, 0.4512, 0.3755, 0.3655, 0.3761, 0.3807], 'Depth': [1, 3, 5, 7, 1, 3, 5, 7] }

f, axes = plt.subplots(1,2)

# Plot the responses for different events and regions
ax = sns.lineplot(x="Depth", y="MSE", hue="Method", data=exp2_mse, ax=axes[1])
ax.set(xlabel='Depth', ylabel='MSE')
ax.set(xticks=(1, 3, 5, 7))
ax.set(title='STL10:5k, ReLU')
plt.ylim(0.00,0.15)
plt.xlim(0.5,7.5)

ax = sns.lineplot(x="Depth", y="Accuracy", hue="Method", data=exp2_mse, ax=axes[0])
ax.set(xlabel='Depth', ylabel='Accuracy')
ax.set(xticks=(1, 3, 5, 7))
ax.set(title='STL10:5k, ReLU')
plt.ylim(0.0,0.6)
plt.xlim(0.5,7.5)
plt.show()
