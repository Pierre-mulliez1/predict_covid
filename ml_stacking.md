---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
#import packages 
import sklearn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
```

```python
#Load dataset

DIRECTORY_WHERE_THIS_FILE_IS = os.path.dirname(os.path.abspath("ml_stacking.md"))
DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "data/prepared_dataset.csv")
df1 = pd.read_csv(DATA_PATH)




```

```python
print(df1.shape)
print(df1.head())

#drop the index 
df1 = df1.drop("Unnamed: 0", axis = 1)
```

```python
#some insights
print("number of provinces: {}".format(df1['provincia'].nunique()))
print(df1.describe())
```

```python
print(df1.head(2))
print('')
print(df2.head(2))
print('')
print(df3.head(2))
print('')
print(df4.head(2))
print('')
print(df5.head(2))
```

```python
df8 = df4.merge(df5,on = "id", how = "left")
df9 =  df2.merge(df3,on = "id", how = "left")
```

```python
print(df8.head())
print('')
df9.head(2)
```

```python
print(df8["jets"].unique())
plt.hist(x=df8["jets"],color = "red")
```

```python
#preprocessin: dumifying 

```

```python
#df8 = pd.get_dummies(df8, columns=[''], prefix = '',drop_first=True)
```

```python
#Null values 
MR = df8['MR'].mean()
e1 = df8['E1'].mean()
```

```python
print(df8.isna().sum())
#replace with mean
df8['MR'] = df8['MR'].replace("",0.0).fillna(MR)
df9['MR'] = df9['MR'].replace("",0.0).fillna(MR)

df8['E1'] = df8['E1'].replace("",0.0).fillna(e1)
df9['E1'] = df9['E1'].replace("",0.0).fillna(e1)
```

```python
pd.set_option('display.max_columns', None)
df8.head()
```

```python

```

```python

```

```python

```

## Machine learning


### Split the dataset

```python
from sklearn.model_selection import train_test_split
```

```python
y = df8["jets"]
X = df8.loc[:,df8.columns != "jets"]
```

```python
#scale 
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X)
```

```python
#remove correlated features
def get_correlation(data, threshold):
    corr_col = set()
    corrmat = data.corr()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if abs(corrmat.iloc[i, j]) > threshold:
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col

corr_features = get_correlation(X, 0.70)
print('correlated features: ', len(set(corr_features)) )
corr_features
```

```python
X = X.drop(labels=corr_features, axis = 1)
```

```python
#train test
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.20, random_state=42)
```

```python
X
```

#### Prepare submission table

```python
df9 = df9.sort_values(by='id')
df9

scaler = StandardScaler().fit(df9)
df9 = scaler.transform(df9)
df9 = pd.DataFrame(df9)
#corr_features = get_correlation(X, 0.70)
df9 = df9.drop(labels = corr_features, axis = 1)
df9
```

## Create dataset for weak learners

```python
weak_leaner = pd.DataFrame()
```

```python
#weak_leaner['targetTRUE'] = df10['jets']
#weak_leaner['ra'] = X_test['ra']
weak_leaner.shape
```

```python
weak_leaner.head()
```

## Random forest 

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV





random_grid = {'n_estimators': [200,300,500,800,1300,1500],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [10,30,50,80],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4]}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)
rf_random.best_params_
```

'n_estimators': 500,
 'min_samples_split': 10,
 'min_samples_leaf': 1,
 'max_features': 'sqrt',
 'max_depth': 80}

```python
# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 500,min_samples_split = 10,
 min_samples_leaf= 1,
 max_features = 'sqrt',
 max_depth = 80)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
  
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
  
# metrics are used to find accuracy or error
from sklearn import metrics  
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
```

```python
weak_leaner['targetrandomF'] = y_pred
```

```python
df1['jets'] = weak_leaner['targetrandomF']
```

## Ridge regression

```python
from sklearn.linear_model import LogisticRegression
```

```python
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
```

```python
weak_leaner['targetridge'] = y_pred
```

```python

```

## SVM 

```python
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```


param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, cv = 3)
  
#### fitting the model for grid search
grid.fit(X_train, y_train)
grid.best_params_

```python
csvm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
csvm.fit(X_train, y_train)


y_pred = csvm.predict(X_test)

print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

```

```python
weak_leaner['targetsvm'] = y_pred
```

```python
df1['jets'] = weak_leaner['targetsvm']
```

## Nearest neighbor

```python
from sklearn.neighbors import NearestCentroid
neigh = NearestCentroid()
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)

print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
```

```python
weak_leaner['targetneigh'] = y_pred
```

```python

```

## PCA LDA 

```python
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
```

```python
X.columns.nunique()
```

```python
pca = PCA(n_components=13) #covariant Matrix
x_pca = pca.fit_transform(X)
df_pca =  pca.fit_transform(df9)
variance = pca.explained_variance_ratio_ #calculate variance ratios
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
print(var)
```

```python
x_pca = pd.DataFrame(x_pca)
x_pca = x_pca.drop(labels = [0,1,2,3,4,5], axis = 1)

```

```python
df_pca  = pd.DataFrame(df_pca)
df_pca = df_pca.drop(labels = [0,1,2,3,4,5], axis = 1)
```

```python
#train test
X_train, X_test, y_train, y_test = train_test_split(
   x_pca, y, test_size=0.20, random_state=42)
```

```python
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
```

```python
y_pred = model.predict(X_test)
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
#weak_leaner['targetlda'] = y_pred
```

## Strong learners

```python
#data 
weak_leaner.head()
```

```python
weak_leaner = pd.get_dummies(weak_leaner, columns=['targetrandomF','targetridge',
                                                   'targetneigh','targetsvm'], prefix = ['random_','ridge_','neigh_','svm_']
                             ,drop_first=True)

```

```python
weak_leaner
```

```python
y = weak_leaner["targetTRUE"]
X = weak_leaner.loc[:,weak_leaner.columns != "targetTRUE"]
```

```python
#train test
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.40, random_state=42)
```

## AdaBoost

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
```

```python
boost = GradientBoostingClassifier(n_estimators=500, learning_rate=0.5,max_depth=50, random_state=0).fit(X_train, y_train)
```

```python
y_pred = boost.predict(X_test)

print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
```

```python

```

## Neural network

```python
from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                  hidden_layer_sizes=(5, 2), random_state=1, max_iter=150)
nn.fit(X_train, y_train)

y_pred = nn.predict(X_test)

print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
```

```python
weak_leaner['targetnn'] = y_pred
df1['jets'] = weak_leaner['targetnn']
```

# Submit

```python
df1.to_csv('submit7.csv',index=False,header = True)
```

```python

```
