import matplotlib.pyplot as plt

# Sample data
altitudes = [10, 20, 30, 40, 50]  # Altitude in meters
detection_accuracy = [0.93, 0.94, 0.95, 0.92, 0.90]  # Accuracy in percentage

plt.figure(figsize=(10, 5))
plt.plot(altitudes, detection_accuracy, marker='o', linestyle='-', color='b', label='Detection Accuracy')
plt.title('Detection Accuracy vs. Altitude')
plt.xlabel('Altitude (m)')
plt.ylabel('Detection Accuracy (%)')
plt.grid(True)
plt.legend()
plt.show()
