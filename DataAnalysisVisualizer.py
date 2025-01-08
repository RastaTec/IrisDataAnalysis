import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load the datasey
url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names=['sepal_length', 'sepal_width', 'petal_length', 'species']
iris=pd.read_csv(url, header=None, names=column_names)

#Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris.head())

#Summary statistics of the dataset
print("\nSummary statistics:")
print(iris.head())

#Count of each species
print("\nCount of each species:")
print(iris['species'].value_counts())

#Pairplot of the dataset
sns.pairplot(iris, hue='species')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.savefig('pairplot.png')
plt.show()

#Correlation matrix heatmap
plt.figure(figsize=(10,8))
sns.heatmap(iris.corr(), annot=True, cmaps='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('heatmap.png')
plt.show()

#Boxplot of features
plt.figure(figsize=(12, 6))
sns.boxplot(data=iris, orients="h", palette="Set2")
plt.title('Boxplot of Iris features')
plt.savefig('boxplot.png')
plt.show()
