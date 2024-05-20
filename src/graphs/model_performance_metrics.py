import matplotlib.pyplot as plt

# Sample data
epochs = [1, 2, 3, 4, 5]
accuracy = [0.85, 0.88, 0.90, 0.92, 0.95]
precision = [0.83, 0.86, 0.88, 0.91, 0.96]
recall = [0.82, 0.85, 0.87, 0.90, 0.94]
f1_scores = [0.82, 0.85, 0.87, 0.90, 0.95]

plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracy, marker='o', linestyle='-', color='b', label='Accuracy')
plt.plot(epochs, precision, marker='o', linestyle='-', color='r', label='Precision')
plt.plot(epochs, recall, marker='o', linestyle='-', color='g', label='Recall')
plt.plot(epochs, f1_scores, marker='o', linestyle='-', color='y', label='F1 Score')
# plt.title('Figure 3: Model Performance Metrics of Maize Disease Detection Model Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.grid(True)
plt.show()
