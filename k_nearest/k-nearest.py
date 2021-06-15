from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

#finding the best value of K with the use of KNeighborsClassifier
breast_cancer_data = load_breast_cancer()
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(k)
  classifier.fit(training_data, training_labels)
  score = classifier.score(validation_data, validation_labels)
  accuracies.append(score)
  
k_list = range(1, 101)

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
#for split with random_state=100 => k = 25

#own implementation of K-Nearest algorithm with the best value of K
#normalization function
def min_max_normalize(array):
  number_columns = len(array[0])
  number_rows = len(array)
  min_column = np.zeros(number_columns)
  max_column = np.zeros(number_columns)
  for i in range(number_columns):
      min_column[i] = min(array[:,i])
      max_column[i] = max(array[:,i])
  normalized = np.zeros(shape=(number_rows, number_columns))
  for row in range(number_rows):
    for col in range(number_columns):
        norm = (array[row][col] - min_column[col])/(max_column[col] - min_column[col])
        normalized[row, col] = norm
  return normalized

#distance calculation
def distance(point1, point2):
  squared_difference = 0
  for i in range(len(point1)):
    squared_difference += (point1[i] - point2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

#classification algorithm
def classify(unknown, dataset, labels, k):
  distances = []
  number_rows = len(dataset)
  for row in range(number_rows):
    point = dataset[row]
    distance_to_point = distance(point, unknown)
    distances.append([distance_to_point, row])
  distances.sort()
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    row = neighbor[1]
    if labels[row] == 0:
      num_bad += 1
    elif labels[row] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0

#classification of validation set
normalized_training = min_max_normalize(training_data)
normalized_validation = min_max_normalize(validation_data)

classified = []
for point in range(len(validation_data)):
    normalized_point = normalized_validation[point]
    classified.append(classify(normalized_point, normalized_training, training_labels, 25))

#calculating accuracy for own algorithm
wrong = 0
correct = 0
for idx in range(len(classified)):
    if classified[idx] == validation_labels[idx]:
        correct += 1
    elif classified[idx] != validation_labels[idx]:
        wrong += 1
print(correct, wrong)
print(1.0 * correct/(correct + wrong))
print(1.0 * wrong/(correct + wrong))

#comparison to sklearn classifier accuracy 
classifier = KNeighborsClassifier(25)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))






