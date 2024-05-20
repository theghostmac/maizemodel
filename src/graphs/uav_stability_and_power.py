import matplotlib.pyplot as plt

# Sample data
conditions = ['Without Payload', 'With Payload']
stability_scores = [9.5, 8.5]
power_consumption = [50, 60]

fig, ax1 = plt.subplots()

ax1.set_xlabel('Conditions')
ax1.set_ylabel('Stability Score', color='tab:blue')
ax1.bar(conditions, stability_scores, color='tab:blue', alpha=0.6, label='Stability Score')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Power Consumption (W)', color='tab:orange')
ax2.plot(conditions, power_consumption, color='tab:orange', marker='o', linestyle='-', label='Power Consumption')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.tight_layout()
plt.title('UAV Stability and Power Consumption')
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.85))
plt.show()
