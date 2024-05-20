import matplotlib.pyplot as plt

# Sample data
phases = ['Images Captured', 'Images Processed', 'Successful Identifications']
counts = [100, 95, 90]

plt.figure(figsize=(10, 5))
plt.bar(phases, counts, color='orange', alpha=0.7)
# plt.title('Model and UAV Synergy: Image Processing Workflow')
plt.xlabel('Phases')
plt.ylabel('Number of Images')
plt.ylim(0, 110)
plt.grid(axis='y')
plt.show()
