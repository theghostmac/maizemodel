import matplotlib.pyplot as plt

# Sample data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
common_rust = [5, 10, 15, 20, 25]
northern_blight = [4, 8, 12, 16, 20]
maize_leaf_spots = [3, 6, 9, 12, 15]

plt.figure(figsize=(10, 5))
plt.plot(months, common_rust, marker='o', linestyle='-', color='r', label='Common Rust')
plt.plot(months, northern_blight, marker='o', linestyle='-', color='g', label='Northern Blight')
plt.plot(months, maize_leaf_spots, marker='o', linestyle='-', color='b', label='Maize Leaf Spots')
# plt.title('Disease Spread Over Time')
plt.xlabel('Months')
plt.ylabel('Number of Cases')
plt.legend()
plt.grid(True)
plt.show()
