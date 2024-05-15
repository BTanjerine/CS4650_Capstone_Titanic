```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.impute import SimpleImputer
```


```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
```


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## Data Preprocessing
- get rid of nulls
- encoding categorical data


```python
train_df.isna().any()
```




    PassengerId    False
    Survived       False
    Pclass         False
    Name           False
    Sex            False
    Age             True
    SibSp          False
    Parch          False
    Ticket         False
    Fare           False
    Cabin           True
    Embarked        True
    dtype: bool




```python
# any one without a cabin just place None
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: 'None' if pd.isna(x) else x)
test_df['Cabin'] = test_df['Cabin'].apply(lambda x: 'None' if pd.isna(x) else x)
```


```python
train_df.isna().any()
```




    PassengerId    False
    Survived       False
    Pclass         False
    Name           False
    Sex            False
    Age             True
    SibSp          False
    Parch          False
    Ticket         False
    Fare           False
    Cabin          False
    Embarked        True
    dtype: bool




```python
# find the most common location embarked
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
```


```python
train_df.isna().any()
```




    PassengerId    False
    Survived       False
    Pclass         False
    Name           False
    Sex            False
    Age             True
    SibSp          False
    Parch          False
    Ticket         False
    Fare           False
    Cabin          False
    Embarked       False
    dtype: bool




```python
# Age is usually correlated with the amount of money they make and sex. 
# find the average age based on sex and passenger class (1st 2nd or 3rd)
corr_df = train_df.groupby(['Pclass', 'Sex']).Age.mean().reset_index()

def update_age(params):
    pclass = params.iloc[0]
    sex = params.iloc[1]
    age = params.iloc[2]
    if pd.isnull(age):
        age = float(corr_df[(corr_df['Pclass'] == pclass) & (corr_df['Sex'] == sex)]['Age'].iloc[0])
    return age

train_df['Age'] = train_df[['Pclass', 'Sex', 'Age']].apply(lambda x : update_age(x), axis = 1)
```


```python
corr_df = test_df.groupby(['Pclass', 'Sex']).Age.mean().reset_index()

test_df['Age'] = test_df[['Pclass', 'Sex', 'Age']].apply(lambda x : update_age(x), axis = 1)
```


```python
train_df.isna().any()
```




    PassengerId    False
    Survived       False
    Pclass         False
    Name           False
    Sex            False
    Age            False
    SibSp          False
    Parch          False
    Ticket         False
    Fare           False
    Cabin          False
    Embarked       False
    dtype: bool




```python
test_df.isna().any()
```




    PassengerId    False
    Pclass         False
    Name           False
    Sex            False
    Age            False
    SibSp          False
    Parch          False
    Ticket         False
    Fare            True
    Cabin          False
    Embarked       False
    dtype: bool




```python
corr_df = train_df.groupby(['Pclass']).Fare.mean().reset_index()

def update_fare(params):
    pclass = params.iloc[0]
    fare = params.iloc[1]
    if pd.isna(fare):
        fare = float(corr_df[(corr_df['Pclass'] == pclass)]['Fare'].iloc[0])
    return fare

test_df['Fare'] = test_df[['Pclass', 'Fare']].apply(lambda x : update_fare(x), axis = 1)
```


```python
test_df.isna().any()
```




    PassengerId    False
    Pclass         False
    Name           False
    Sex            False
    Age            False
    SibSp          False
    Parch          False
    Ticket         False
    Fare           False
    Cabin          False
    Embarked       False
    dtype: bool




```python
# get rid of uneeded columns 
train_df = train_df.drop(columns=['Name', 'Ticket', 'Cabin'])
test_df = test_df.drop(columns=['Name', 'Ticket', 'Cabin'])
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## Data Analysis


```python
sns.countplot(x='Sex', hue='Survived', data=train_df)
plt.title('Survival by Gender')
plt.show()
```


    
![png](capstone_project_files/capstone_project_17_0.png)
    



```python
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title('Survival by Passenger Class')
plt.show()
```


    
![png](capstone_project_files/capstone_project_18_0.png)
    



```python
plt.figure(figsize=(10, 6))
sns.histplot(data=train_df, x='Age', bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
```


    
![png](capstone_project_files/capstone_project_19_0.png)
    



```python
train_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.318643</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>13.281103</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>21.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>26.507589</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>36.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df[['Age', 'Pclass']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Age', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>38.378866</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>29.907295</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>25.112288</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.629630</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.472826</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.242363</td>
    </tr>
  </tbody>
</table>
</div>




```python
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
```




    <seaborn.axisgrid.FacetGrid at 0x1708ca62d20>




    
![png](capstone_project_files/capstone_project_23_1.png)
    


## One hot encode categorical data 
- take text data like Sex and Embarked and split them into binary columns of different cateories
- aka one hot encoding


```python
# one hot encode sex and embarked columns
encoder = OneHotEncoder()
sex_arr = encoder.fit_transform(train_df[['Sex']]).toarray()

# split sex column into male and female columns
categories = ['female', 'male']
for i in range(len(sex_arr.T)):
    train_df[categories[i]] = sex_arr.T[i]

sex_arr = encoder.fit_transform(test_df[['Sex']]).toarray()

# split sex column into male and female columns
categories = ['female', 'male']
for i in range(len(sex_arr.T)):
    test_df[categories[i]] = sex_arr.T[i]

```


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>female</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
emb_arr = encoder.fit_transform(train_df[['Embarked']]).toarray()

# split sex column into male and female columns
categories = ['C','S','Q','N']
for i in range(len(emb_arr.T)):
    train_df[categories[i]] = emb_arr.T[i]

emb_arr = encoder.fit_transform(test_df[['Embarked']]).toarray()

# split sex column into male and female columns
categories = ['C','S','Q','N']
for i in range(len(emb_arr.T)):
    test_df[categories[i]] = emb_arr.T[i]
```


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>S</th>
      <th>Q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remove uneeded columns
train_df = train_df.drop(columns=['Sex', 'Embarked'])
test_df = test_df.drop(columns=['Sex', 'Embarked'])
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>S</th>
      <th>Q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>S</th>
      <th>Q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>34.500000</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>47.000000</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>62.000000</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>27.000000</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>22.000000</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1305</td>
      <td>3</td>
      <td>24.525104</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>1</td>
      <td>39.000000</td>
      <td>0</td>
      <td>0</td>
      <td>108.9000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>3</td>
      <td>38.500000</td>
      <td>0</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>3</td>
      <td>24.525104</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>3</td>
      <td>24.525104</td>
      <td>1</td>
      <td>1</td>
      <td>22.3583</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 11 columns</p>
</div>



## Normalize Data Points
- use standard scaler to make 0 the mean and 1 the standard deviation


```python
def normalize_data(df):
    numeric_data = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(numeric_data)
    norm_df = pd.DataFrame(norm_data, columns=numeric_data.columns)
    df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']] = norm_df

normalize_data(train_df)
normalize_data(test_df)
```


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>S</th>
      <th>Q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0.827377</td>
      <td>-0.551366</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>-0.502445</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>-1.566107</td>
      <td>0.654030</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.786845</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>0.827377</td>
      <td>-0.250017</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.488854</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>-1.566107</td>
      <td>0.428018</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>0.420730</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0.827377</td>
      <td>0.428018</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.486337</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>S</th>
      <th>Q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0.873482</td>
      <td>0.392121</td>
      <td>-0.499470</td>
      <td>-0.400248</td>
      <td>-0.497374</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0.873482</td>
      <td>1.357557</td>
      <td>0.616992</td>
      <td>-0.400248</td>
      <td>-0.512238</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>-0.315819</td>
      <td>2.516080</td>
      <td>-0.499470</td>
      <td>-0.400248</td>
      <td>-0.464061</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0.873482</td>
      <td>-0.187140</td>
      <td>-0.499470</td>
      <td>-0.400248</td>
      <td>-0.482436</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0.873482</td>
      <td>-0.573315</td>
      <td>0.616992</td>
      <td>0.619896</td>
      <td>-0.417453</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# Logistic Regression Model


```python
# setup data
X = train_df.iloc[:, 2:]
Y = train_df.iloc[:, 1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8)

x_eval = test_df.iloc[:, 1:]
```


```python
log_mod = LogisticRegression()

log_mod.fit(x_train, y_train)

y_pred = log_mod.predict(x_test)
```


```python
print(f"accuracy of training set: {100*log_mod.score(x_train,y_train):.2f}%")
print(f"accuracy of training set: {100*log_mod.score(x_test,y_test):.2f}%")
```

    accuracy of training set: 80.76%
    accuracy of training set: 85.47%
    


```python
actual = y_test
predicted = y_pred

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()
```


    
![png](capstone_project_files/capstone_project_39_0.png)
    



```python
print(metrics.classification_report(y_test, y_pred, target_names={'Did not Survive', 'Survived'}))
```

                     precision    recall  f1-score   support
    
           Survived       0.85      0.93      0.89       110
    Did not Survive       0.86      0.74      0.80        69
    
           accuracy                           0.85       179
          macro avg       0.86      0.83      0.84       179
       weighted avg       0.86      0.85      0.85       179
    
    


```python
y_pred_proba = log_mod.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.title("ROC Curve")
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
```


    
![png](capstone_project_files/capstone_project_41_0.png)
    



```python
def print_prediction(pred):
    if pred[0] == 1:
        print("This person probably survived")
    else:
        print("This person probably did not survive")

# P_class:3	Age: 27.0 sibsp:0 parch:0 Fare: 8.6625 Sex: Male Embarked to: Q
prediction_eval = log_mod.predict(x_eval.iloc[[3]])
print_prediction(prediction_eval)
```

    This person probably did not survive
    


```python
# P_class:1	Age: 39.0 sibsp:0 parch:0 Fare: 108.90 Sex: Female Embarked to: C
prediction_eval = log_mod.predict(x_eval.iloc[[414]])
print_prediction(prediction_eval)
```

    This person probably survived
    

## Get Evaluation data results into csv file


```python
prediction_eval = log_mod.predict(x_eval)
```


```python
result_data = pd.DataFrame(test_df['PassengerId'], columns=['PassengerId'])
result_data = result_data.assign(Survived=prediction_eval)
result_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
result_data.to_csv("titanic_submission.csv", index=False)
!cat titanic_submission.csv
```

    PassengerId,Survived
    892,0
    893,0
    894,0
    895,0
    896,0
    897,0
    898,1
    899,0
    900,1
    901,0
    902,0
    903,0
    904,1
    905,0
    906,1
    907,1
    908,0
    909,0
    910,0
    911,0
    912,0
    913,0
    914,1
    915,1
    916,1
    917,0
    918,1
    919,0
    920,0
    921,0
    922,0
    923,0
    924,0
    925,0
    926,0
    927,0
    928,1
    929,1
    930,0
    931,0
    932,0
    933,0
    934,0
    935,1
    936,1
    937,0
    938,0
    939,0
    940,1
    941,0
    942,1
    943,0
    944,1
    945,1
    946,0
    947,0
    948,0
    949,0
    950,0
    951,1
    952,0
    953,0
    954,0
    955,1
    956,1
    957,1
    958,1
    959,0
    960,1
    961,1
    962,1
    963,0
    964,1
    965,1
    966,1
    967,1
    968,0
    969,1
    970,0
    971,1
    972,0
    973,0
    974,0
    975,0
    976,0
    977,0
    978,1
    979,1
    980,1
    981,0
    982,1
    983,0
    984,1
    985,0
    986,1
    987,0
    988,1
    989,0
    990,1
    991,0
    992,1
    993,0
    994,0
    995,0
    996,1
    997,0
    998,0
    999,0
    1000,0
    1001,0
    1002,0
    1003,1
    1004,1
    1005,1
    1006,1
    1007,0
    1008,0
    1009,1
    1010,1
    1011,1
    1012,1
    1013,0
    1014,1
    1015,0
    1016,0
    1017,1
    1018,0
    1019,0
    1020,0
    1021,0
    1022,0
    1023,0
    1024,1
    1025,0
    1026,0
    1027,0
    1028,0
    1029,0
    1030,1
    1031,0
    1032,0
    1033,1
    1034,0
    1035,0
    1036,0
    1037,0
    1038,0
    1039,0
    1040,0
    1041,0
    1042,1
    1043,0
    1044,0
    1045,0
    1046,0
    1047,0
    1048,1
    1049,1
    1050,0
    1051,1
    1052,1
    1053,0
    1054,1
    1055,0
    1056,0
    1057,0
    1058,0
    1059,0
    1060,1
    1061,1
    1062,0
    1063,0
    1064,0
    1065,0
    1066,0
    1067,1
    1068,1
    1069,0
    1070,1
    1071,1
    1072,0
    1073,0
    1074,1
    1075,0
    1076,1
    1077,0
    1078,1
    1079,0
    1080,0
    1081,0
    1082,0
    1083,0
    1084,0
    1085,0
    1086,0
    1087,0
    1088,1
    1089,1
    1090,0
    1091,1
    1092,1
    1093,0
    1094,0
    1095,1
    1096,0
    1097,0
    1098,1
    1099,0
    1100,1
    1101,0
    1102,0
    1103,0
    1104,0
    1105,0
    1106,0
    1107,0
    1108,1
    1109,0
    1110,1
    1111,0
    1112,1
    1113,0
    1114,1
    1115,0
    1116,1
    1117,1
    1118,0
    1119,1
    1120,0
    1121,0
    1122,0
    1123,1
    1124,0
    1125,0
    1126,0
    1127,0
    1128,0
    1129,0
    1130,1
    1131,1
    1132,1
    1133,1
    1134,0
    1135,0
    1136,0
    1137,0
    1138,1
    1139,0
    1140,1
    1141,1
    1142,1
    1143,0
    1144,1
    1145,0
    1146,0
    1147,0
    1148,0
    1149,0
    1150,1
    1151,0
    1152,0
    1153,0
    1154,1
    1155,1
    1156,0
    1157,0
    1158,0
    1159,0
    1160,1
    1161,0
    1162,0
    1163,0
    1164,1
    1165,1
    1166,0
    1167,1
    1168,0
    1169,0
    1170,0
    1171,0
    1172,1
    1173,0
    1174,1
    1175,1
    1176,1
    1177,0
    1178,0
    1179,1
    1180,0
    1181,0
    1182,0
    1183,1
    1184,0
    1185,0
    1186,0
    1187,0
    1188,1
    1189,0
    1190,1
    1191,0
    1192,0
    1193,0
    1194,0
    1195,0
    1196,1
    1197,1
    1198,0
    1199,0
    1200,0
    1201,0
    1202,0
    1203,0
    1204,0
    1205,0
    1206,1
    1207,1
    1208,0
    1209,0
    1210,0
    1211,0
    1212,0
    1213,0
    1214,0
    1215,1
    1216,1
    1217,0
    1218,1
    1219,0
    1220,0
    1221,0
    1222,1
    1223,0
    1224,0
    1225,1
    1226,0
    1227,1
    1228,0
    1229,0
    1230,0
    1231,0
    1232,0
    1233,0
    1234,0
    1235,1
    1236,0
    1237,1
    1238,0
    1239,0
    1240,0
    1241,1
    1242,1
    1243,0
    1244,0
    1245,0
    1246,1
    1247,0
    1248,1
    1249,0
    1250,0
    1251,0
    1252,0
    1253,1
    1254,1
    1255,0
    1256,1
    1257,0
    1258,0
    1259,1
    1260,1
    1261,0
    1262,0
    1263,1
    1264,0
    1265,0
    1266,1
    1267,1
    1268,0
    1269,0
    1270,0
    1271,0
    1272,0
    1273,0
    1274,1
    1275,1
    1276,0
    1277,1
    1278,0
    1279,0
    1280,0
    1281,0
    1282,1
    1283,1
    1284,0
    1285,0
    1286,0
    1287,1
    1288,0
    1289,1
    1290,0
    1291,0
    1292,1
    1293,0
    1294,1
    1295,1
    1296,0
    1297,0
    1298,0
    1299,0
    1300,1
    1301,1
    1302,1
    1303,1
    1304,1
    1305,0
    1306,1
    1307,0
    1308,0
    1309,0
    


```python

```
