import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
data = digits.data

#choosing k - elbow method
num_clusters = list(range(1, 20))
inertias = []

for i in num_clusters:
  model = KMeans(n_clusters = i)
  model.fit(data)
  inertias.append(model.inertia_)
  
plt.plot(num_clusters, inertias, '-o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

#final model
k = 10
model = KMeans(n_clusters = i)
model.fit(data)

