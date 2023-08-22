import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from FS.pso import jfs  # Change this to switch algorithm
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('dataset/lwk_ctr.csv')
feat = data.iloc[:, :-1].values
label = data.iloc[:, -1].values

# Split data into train & validation (70-30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}

# Parameters
k = 5  # k-value in KNN
N = 10  # number of particles
T = 100  # maximum number of iterations
opts = {'k': k, 'fold': fold, 'N': N, 'T': T}

# Perform feature selection
fmdl = jfs(feat, label, opts)
sf = fmdl['sf']

# Get selected feature names
selected_feature_names = data.columns[sf]  # Assuming 'data' is your DataFrame
# print("Selected Feature Names:", selected_feature_names)

# Model with selected features
x_train = xtrain[:, sf]
x_valid = xtest[:, sf]
mdl = KNeighborsClassifier(n_neighbors=k)
mdl.fit(x_train, ytrain)

# Accuracy
y_pred = mdl.predict(x_valid)
accuracy = np.sum(ytest == y_pred) / len(ytest)
print("Accuracy:", 100 * accuracy)

# Number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)
print("Selected Features:", sf)
print("Selected Feature Names:", selected_feature_names)

# Plot convergence
curve = fmdl['c']
x = np.arange(1, T + 1)
curve = curve.reshape(-1)  # Reshape curve array
fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('PSO')
ax.grid()
plt.show()