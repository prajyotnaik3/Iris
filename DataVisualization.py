import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


dataset = pd.read_csv(r"data\iris.csv")
#print(dataset.head())
#print(dataset.describe())
#print(dataset.columns)
print(dataset['class'].value_counts())

for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    print('\n*** Results for {} ***'.format(feature))
    print(dataset.groupby('class')[feature].describe())
    versicolor = dataset[dataset['class']=='Iris-versicolor'][feature]
    setosa = dataset[dataset['class']=='Iris-setosa'][feature]
    virginia = dataset[dataset['class']=='Iris-virginica'][feature]
    

for i in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    versicolor = dataset[dataset['class']=='Iris-versicolor'][i]
    setosa = dataset[dataset['class']=='Iris-setosa'][i]
    virginia = dataset[dataset['class']=='Iris-virginica'][i]
    xmin = min(min(versicolor), min(setosa), min(virginia))
    xmax = max(max(versicolor), max(setosa), max(virginia))
    width = (xmax - xmin) / 40
    sns.distplot(versicolor, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(setosa, color='g', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(virginia, color='b', kde=False, bins=np.arange(xmin, xmax, width))
    plt.legend(['versicolor', 'setosa', 'virginia'])
    plt.title('Overlaid histogram for {}'.format(i))
    plt.show()
    
for feat in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    outliers = []
    data = dataset[feat]
    mean = np.mean(data)
    std = np.std(data)
    for y in data:
        z_score= (y - mean)/std 
        if np.abs(z_score) > 3:
            outliers.append(y)
    print('\nOutlier caps for {}:'.format(feat))
    print('  --95p: {:.1f} / {} values exceed that'.format(data.quantile(.95), 
          len([i for i in data if i > data.quantile(.95)])))
    print('  --3sd: {:.1f} / {} values exceed that'.format(mean + 3*(std), len(outliers)))
    print('  --99p: {:.1f} / {} values exceed that'.format(data.quantile(.99),
          len([i for i in data if i > data.quantile(.99)])))
dataset['sepal_length_clean'] = dataset['sepal_length'].clip(upper = dataset['sepal_length'].quantile(.99))
dataset['sepal_width_clean'] = dataset['sepal_width'].clip(upper = dataset['sepal_width'].quantile(.99))
dataset['petal_length_clean'] = dataset['petal_length'].clip(upper = dataset['petal_length'].quantile(.99))

#Check for skewness in features
for feature in ['sepal_length_clean', 'sepal_width_clean', 'petal_length_clean', 'petal_width']:
    sns.distplot(dataset[feature], kde=False)
    plt.title('Histogram for {}'.format(feature))
    plt.show()
    
dataset.to_csv(r"data\iris_clean.csv", index = False)