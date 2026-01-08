import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('suman_results.csv')

# Clean the data
# Remove units and convert to numeric
df['Inference Time Value'] = df['INFERENCE TIME'].astype(str).str.replace(' ms', '').astype(float)
df['FPS Value'] = df['FPS'].astype(str).str.replace(' FPS', '').astype(float)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot Inference Time
bars1 = ax1.bar(df['MODEL NAME'], df['Inference Time Value'], color='skyblue')
ax1.set_title('Inference Time per Model')
ax1.set_xlabel('Model Name')
ax1.set_ylabel('Inference Time (ms)')
ax1.tick_params(axis='x', rotation=45)
# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

# Plot FPS
bars2 = ax2.bar(df['MODEL NAME'], df['FPS Value'], color='salmon')
ax2.set_title('Frames Per Second (FPS) per Model')
ax2.set_xlabel('Model Name')
ax2.set_ylabel('FPS')
ax2.tick_params(axis='x', rotation=45)
# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_performance_plot.png')