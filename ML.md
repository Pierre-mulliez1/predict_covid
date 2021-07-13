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
print("Number of provinces: {}".format(df1['provincia'].nunique()))
print("Approximate number of days by region: {}".format(df1['fecha'].nunique()))
#print("Number of distinct age groups: {}".format(df1['grupo_edad'].nunique()))
print('')
print(517*51*10)
print(df1.describe())
```

## Machine learning


### Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df1.columns
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
```

```python
#divide by countries ! -> no dummies 
def Prepare_dataset(df = df1,target = "num_casos",test = 0.2,dimension_reduction = False,scale = True, dummies = 'e'):
    
    #drop the province name and date
    df = df.drop(labels=['provincia','fecha','date','poblacion'], axis = 1)
    
    #scale ( not the dummies or target)
    col = ['provincia_iso','Communidad','grupo_edad','sexo','year','dayyear',target]

  
    if scale == True:
        col_s = ['num_casos', 'num_casos_prueba_pcr',
       'num_casos_prueba_test_ac', 'num_casos_prueba_ag',
       'num_casos_prueba_elisa', 'num_casos_prueba_desconocida', 'people_fully_vaccinated_per_hundred', 'France_cases_mil',
       'Portugal_cases_mil']
        S = pd.DataFrame(df.loc[:,col_s])
        scaler = StandardScaler().fit(S)
        S = pd.DataFrame(scaler.transform(pd.DataFrame(S)))
        df = pd.concat([df, S], axis=1)
    
    
    #dumify state and regions  
    if dummies != 'e':
        df = pd.get_dummies(df, columns=dummies, prefix = ['province_','communidad_'],drop_first=True)
    
    
    #x y split
    y = df[target]
    X = df.loc[:,df.columns != target]
    
    
    if dimension_reduction == False:
        #delete highly correlated features
        corr_features = get_correlation(X, 0.80)
        X = X.drop(labels=corr_features, axis = 1)
    else:
        #PCA dimension reduction
        pca = PCA(n_components= len(X.columns) ) #covariant Matrix
        x_pca = pca.fit_transform(X)
        variance = pca.explained_variance_ratio_ #calculate variance ratios
        var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=5)*100)
        x_pca = pd.DataFrame(x_pca)
        for el in range(0,len(var)):
            if var[el] < 65:
                X = x_pca.drop(labels = el, axis = 1)
    #train test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test, random_state=42)
    
    #prepare weak learner dataset
    weak_leaner = pd.DataFrame()
    weak_leaner['targetTRUE'] = df[target]
    return X_train, X_test, y_train, y_test,weak_leaner
```

```python
Prepare_dataset(dimension_reduction= True, dummies = ['provincia_iso','Communidad'])
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
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
