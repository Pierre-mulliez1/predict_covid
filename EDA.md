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
#First file 
import sklearn
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import datetime as dt
#a
```

```python
# Load files
DIRECTORY_WHERE_THIS_FILE_IS = os.path.dirname(os.path.abspath("EDA"))
DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "data/location.csv")
df1 = pd.read_csv(DATA_PATH, parse_dates=["fecha"])

DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "data/demographics.csv")
df2 = pd.read_csv(DATA_PATH, parse_dates=["fecha"])

#check origin file
#DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "data/movement.xls")
#df3 = pd.read_excel(DATA_PATH)
```

```python
df1
```

```python
#Convert to day of the year
df1['dayyear'] = df1['fecha']
print(df1.shape)
for el in range(0, len(df1['fecha'])):
    day_of_year =  pd.to_datetime(df1.iloc[el,8]).timetuple().tm_yday
    df1.iloc[el,8] = day_of_year
    if el == 15000:
        print("processing halfway")
#dummy collumn for year 
df1['year'] = pd.DatetimeIndex(df1['fecha']).year
```

```python
#Merging
complete_df = pd.merge(df1, df2, how='left', on=['provincia_iso','fecha'])
```

```python
#remov dupl col
#complete_df= complete_df.drop(['num_casos_y'], axis=1)
```

```python
complete_df.tail(5)
```

```python
complete_df.describe()
```

```python
print("There are {} different provinces".format(len(complete_df['provincia_iso'].unique())))


#Navarra code problem
complete_df.loc[complete_df['provincia_iso'].isna() == True,:]
```

```python
complete_df.loc[:,'provincia_iso'] = complete_df.loc[:,'provincia_iso'].replace("","NA").fillna("NA")
```

```python
print(complete_df['provincia_iso'].unique())
complete_df.loc[complete_df['provincia_iso'].isna() == True,:]['provincia_iso']
```

```python
province_cases = complete_df.groupby('provincia_iso').agg({"num_casos_x":np.sum})
date_cases =  complete_df.groupby('fecha').agg({"num_casos_x":np.sum})
age_cases = complete_df.groupby('grupo_edad').agg({"num_casos_y":np.sum})
```

```python
date_cases.plot(figsize = (15,10),ylabel = "number of cases in millions")
```

```python
province_cases.reset_index(inplace=True)
```

```python
plt.figure()
plt.rcParams.update({'font.size': 22})
province_cases.plot.bar(x="provincia_iso", y='num_casos_x', ylabel = "number of cases in millions",rot=70, 
                        fontsize= 16,grid=True,figsize = (20,10),title="Cases by province");
print("Note that cases are not unique")
```

```python
age_cases.reset_index(inplace=True)
```

```python
age_cases.plot.bar(x="grupo_edad", y='num_casos_y', ylabel = "number of cases",rot=70, 
                        fontsize= 16,grid=True,figsize = (20,10),title="Cases by age groups",color = "red");
```

```python
print("data source do not agree with each other")
complete_df.loc[complete_df["num_casos_y"] > 0,:]
```

```python
decision = False 
```

```python
if decision == True:
    #avoid duplicates with age and gender
    complete_df = df1
    complete_df.loc[:,'provincia_iso'] = complete_df.loc[:,'provincia_iso'].replace("","NA").fillna("NA")
```

### Overview of spain and neighboring countries demographic and covid over time 

```python
DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "data/world_cov_data.csv")
df5 = pd.read_csv(DATA_PATH)
```

```python
#Filtering fro a spain and neighboring countries
df5 = df5.loc[(df5["location"] == "Spain") | (df5["location"] == "France") | (df5["location"] == "Portugal"),:]
```

```python
df5["location"].unique()
```

```python
#replacing empty and null values
df5 = df5.replace("",0.0).fillna(0.0)
#delete unused columns
df5 = df5.drop("continent", axis = 1)
```

```python
pd.set_option('display.max_columns', None)
df5.head()
```

```python
df5.columns
```

```python
df5_spain =  df5.loc[(df5["location"] == "Spain"),:]
df5_france = df5.loc[(df5["location"] == "France"),["date","total_cases_per_million"]]
df5_portugal = df5.loc[(df5["location"] == "Portugal"),["date","total_cases_per_million"]]

df5_france.columns = ['date','France_cases_mil']
df5_portugal.columns = ['date','Portugal_cases_mil']
```

```python
#Get the correct date format
df5_spain.iloc[:,2] = pd.to_datetime(df5_spain.iloc[:,2],format="%d/%m/%Y")
df5_france.iloc[:,0] = pd.to_datetime(df5_france.iloc[:,0],format="%d/%m/%Y")
df5_portugal.iloc[:,0] = pd.to_datetime(df5_portugal.iloc[:,0],format="%d/%m/%Y")
```

```python
df5_spain = df5_spain.sort_values(by='date')
```

```python
df5_spain.tail()
```

```python
#delete unused columns
df_s = df5_spain.drop(["iso_code","location","new_cases_smoothed","new_deaths_smoothed","new_cases_smoothed_per_million","new_deaths_smoothed_per_million","weekly_icu_admissions","weekly_icu_admissions_per_million","weekly_hosp_admissions"
                                 ,"new_tests_smoothed","new_tests_smoothed_per_thousand","new_vaccinations_smoothed","new_vaccinations_smoothed_per_million","stringency_index"], axis = 1)
df_s
#we need state values
df_s = df_s.iloc[:,0:29]
```

```python
#Merge the case sof neighboring countries 
global_covid = df_s.merge(df5_france, on = "date", how = "left").merge(df5_portugal, on = "date", how = "left")
#replacing empty and null values
global_covid = global_covid.replace("",0.0).fillna(0.0)
```

```python
#delete duplicate columns with original dataset
gcovid = global_covid.iloc[:,28:]
```

```python
gcovid['date'] = global_covid['date'] 
```

```python
#Merge the case sof neighboring countries 
final_data = complete_df.merge(gcovid, left_on = "fecha", right_on = "date", how = "left")
final_data = final_data.dropna()
```

### Calendar events 

```python
DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "data/holiday_spain.csv")
df6 = pd.read_csv(DATA_PATH)
```

```python
df6 = df6.loc[df6["country"] == "Spain",["ds_holidays","holiday"]]
df6
```

```python

```

### Other disease data

```python
DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "data/other_deseases.xlsx")
df7 = pd.read_excel(DATA_PATH)
```

```python
df7
```

```python

```

### Population by state

```python
DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "data/iso_provinces.xlsx")
df8 = pd.read_excel(DATA_PATH)
df8.loc[:,'provincia_iso'] = df8.loc[:,'provincia_iso'].replace("","NA").fillna("NA")
print(df8.head())
df8['provincia_iso'].unique()
```

```python
final_data1 = final_data.merge(df8, on = "provincia_iso", how = "left")
final_data1 = final_data1.dropna()
```

```python
#Check that all provinces are here 
print("number of provinces: {}".format(final_data1['provincia'].nunique()))
```

```python
#####to csv final dataset
final_data1.to_csv('data/prepared_dataset.csv')
```

```python
final_data1
```

```python
final_data1.loc[final_data1['provincia_iso'] == "NA",:]
```

```python

```
