import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('dataset.csv')

# Calculate the range for binning
min_pka = df['pka_value'].min()
max_pka = df['pka_value'].max()

# Create the histogram with 12 bins within the range
plt.figure(figsize=(10, 6))
plt.hist(df['pka_value'], bins=12, range=(min_pka, max_pka), edgecolor='black')
plt.title('Distribution of pKa Values')
plt.xlabel('pKa Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

# Add text to show the actual range of pKa values
plt.text(0.05, 0.95, f'Range: {min_pka:.2f} - {max_pka:.2f}', 
         transform=plt.gca().transAxes, 
         verticalalignment='top')

# Save the plot
plt.savefig('pka_value_histogram.png')
plt.close()

# Print descriptive statistics
print("pKa Value Statistics:")
print(df['pka_value'].describe())

# Print the bin details
print("\nHistogram Bin Details:")
hist, bin_edges = np.histogram(df['pka_value'], bins=12, range=(min_pka, max_pka))
print("Bin Edges:", bin_edges)
print("Frequencies:", hist)
