import matplotlib.pyplot as plt

# Sample data
image_sizes = ['320x320', '416x416', '512x512', '608x608']
inference_time = [0.05, 0.07, 0.10, 0.15]  # Inference time in seconds

plt.figure(figsize=(10, 5))
plt.bar(image_sizes, inference_time, color='purple', alpha=0.7)
plt.title('Inference Time vs. Image Size')
plt.xlabel('Image Size')
plt.ylabel('Inference Time (s)')
plt.grid(axis='y')
plt.show()
