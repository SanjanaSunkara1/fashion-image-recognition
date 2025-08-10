import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# 8<------------ cut here ---------------
# Load the training dataset
train_data = pd.read_csv('fashion_mnist_20bal_train.csv')

#  ADDED: Filter training data to only include dresses (3) and sneakers (7)
#selected_labels = [3, 5, 7]
#is_label_selected_train = #train_data['class'].isin(selected_labels)
#train_data = train_data[is_label_selected_train]

# Separate the data (features) and the  classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']  # Target (first column)

# Split training data into training and validation sets
X_train, X_validate, y_train, y_validate = train_test_split(
    X_train, y_train, test_size=20, random_state=2, stratify=y_train)


# Load the testing dataset
test_data = pd.read_csv('fashion_mnist_20bal_test.csv')


# ADDED: Filter test data to only include dresses (3) and sneakers (7)
#is_label_selected_test = #test_data['class'].isin(selected_labels)
#test_data = test_data[is_label_selected_test]


# Separate the data (features) and the  classes
X_test = test_data.drop('class',
                        axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']  # Target (first column)


# 8<------------ cut here ---------------
neural_net_model = MLPClassifier(hidden_layer_sizes=(8),random_state=42,tol=0.005)


# 8<------------ cut here ---------------
neural_net_model.fit(X_train, y_train)
# Determine model architecture
layer_sizes = [neural_net_model.coefs_[0].shape[0]
               ]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_
                ]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Layer sizes: {layer_size_str}")


# 8<------------ cut here ---------------
# predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_validate)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
for true, pred in zip(y_validate, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1


# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_validate) * 100
print(f"Overall Validation Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training * 100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")


# Display confusion matrix
conf_matrix = confusion_matrix(y_validate, y_pred)
class_ids = sorted(total_counts.keys())

print("Confusion Matrix:")
print(f"{'':9s}", end='')
for label in class_ids:
    print(f"Class {label:2d} ", end='')
print()

for i, row in enumerate(conf_matrix):
    print(f"Class {class_ids[i]}:", " ".join(f"{num:8d}" for num in row))
