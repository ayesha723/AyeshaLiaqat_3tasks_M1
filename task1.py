import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV using absolute path
data = pd.read_csv(r"F:\artificial intelligence\DEV INTERN M1\task1\Iris.csv")

# Show first 5 rows
print(data.head())

# Check if file is found in current directory (this will still return False because you're using full path)
print("File found?", os.path.exists("Iris.csv"))

# Drop the 'Id' column (capital I)
data = data.drop(columns=['Id'])

# Rename columns to 0,1,2,3 for numerical features
cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
data.rename(columns={cols[0]: 0, cols[1]: 1, cols[2]: 2, cols[3]: 3}, inplace=True)

# Display every 50th row to sample data
print(data.loc[::50])
# Basic info
print(data.shape)
print(data.describe())

# Species count
print(data['Species'].value_counts())

# Plot individual histograms
plt.figure(figsize=(10, 6))
plt.hist(data[0], alpha=0.6, label=cols[0])
plt.hist(data[1], alpha=0.6, label=cols[1])
plt.hist(data[2], alpha=0.6, label=cols[2])
plt.hist(data[3], alpha=0.6, label=cols[3])
plt.legend()
plt.title("Overlapping Histograms of All Features")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Plot 2 histograms (2 and 3) separately with transparency
plt.figure()
plt.hist(data[2], alpha=0.6, label=cols[2])
plt.hist(data[3], alpha=0.5, label=cols[3])
plt.legend()
plt.title("Histogram: Petal Length vs Petal Width")
plt.show()

# Subplots: 4 histograms in grid
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax[0, 0].hist(data[0])
ax[0, 0].set_title(cols[0])

ax[0, 1].hist(data[1])
ax[0, 1].set_title(cols[1])

ax[1, 0].hist(data[2])
ax[1, 0].set_title(cols[2])

ax[1, 1].hist(data[3])
ax[1, 1].set_title(cols[3])

plt.tight_layout()
plt.show()

# Define color mapping for species
colors = {
    'Iris-setosa': 'red',
    'Iris-versicolor': 'green',
    'Iris-virginica': 'blue'
}

# Additional scatter plots
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data[0], data[2], c=data['Species'].map(colors))
plt.xlabel(cols[0])
plt.ylabel(cols[2])
plt.title("Sepal Length vs Petal Length")

plt.subplot(1, 2, 2)
plt.scatter(data[1], data[3], c=data['Species'].map(colors))
plt.xlabel(cols[1])
plt.ylabel(cols[3])
plt.title("Sepal Width vs Petal Width")

plt.tight_layout()
plt.show()

# Correlation matrix
print("Correlation matrix:")
print(data[[0, 1, 2, 3]].corr())


# Simple boxplot of all features
plt.boxplot([data[0], data[1], data[2], data[3]])
plt.xticks([1, 2, 3, 4], cols)
plt.title("Boxplot of All Features")
plt.show()

# Boxplot by species for one column
data.boxplot(column=[0], by=['Species'])
plt.title(f"Boxplot of {cols[0]} by Species")
plt.suptitle("")
plt.show()

# 2x2 boxplots per feature, split by species
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
A = [data[0][data.Species == 'Iris-setosa'], data[0][data.Species == 'Iris-virginica'], data[0][data.Species == 'Iris-versicolor']]
B = [data[1][data.Species == 'Iris-setosa'], data[1][data.Species == 'Iris-virginica'], data[1][data.Species == 'Iris-versicolor']]
C = [data[2][data.Species == 'Iris-setosa'], data[2][data.Species == 'Iris-virginica'], data[2][data.Species == 'Iris-versicolor']]
D = [data[3][data.Species == 'Iris-setosa'], data[3][data.Species == 'Iris-virginica'], data[3][data.Species == 'Iris-versicolor']]

ax[0, 0].boxplot(A, widths=0.7)
ax[0, 0].set_title(cols[0])

ax[0, 1].boxplot(B, widths=0.7)
ax[0, 1].set_title(cols[1])

ax[1, 0].boxplot(C, widths=0.7)
ax[1, 0].set_title(cols[2])

ax[1, 1].boxplot(D, widths=0.7)
ax[1, 1].set_title(cols[3])

plt.tight_layout()
plt.show()

# Colorful combined boxplots for all features
def set_color(bp):
    color_list = ['red', 'green', 'blue']
    for patch, color in zip(bp['boxes'], color_list):
        plt.setp(patch, color=color)

fig = plt.figure(figsize=(10, 6))
bp = plt.boxplot(A, 0, '', positions=[1, 2, 3], widths=0.6)
set_color(bp)
bp = plt.boxplot(B, 0, '', positions=[5, 6, 7], widths=0.6)
set_color(bp)
bp = plt.boxplot(C, 0, '', positions=[9, 10, 11], widths=0.6)
set_color(bp)
bp = plt.boxplot(D, 0, '', positions=[13, 14, 15], widths=0.6)
set_color(bp)

# X-axis labels
plt.xticks([2, 6, 10, 14], cols)
plt.title("Colored Grouped Boxplots of All Features by Species")
plt.show()
