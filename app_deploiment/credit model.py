#!/usr/bin/env python
# coding: utf-8

# In[4]:


# importer les package
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import train_test_split   le plus utiliser de facon classique
 #on a importer les trois algorithm de classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
 #importer une metrique
from sklearn.metrics import accuracy_score
import pickle
import copy


# In[5]:


os.getcwd()


# In[6]:


os.chdir('C:\\Users\\E682\\Desktop\\Projects\\python\\realisez un projet datascience de A a Z\\assets\\file')


# In[7]:


# Importez le fichier de données “coeur” dans dans votre notebook à l’aide de pandas et stockez le dans un objet appelé “data” .
data = pd.read_csv(r'C:\Users\E682\Desktop\Projects\python\realisez un projet datascience de A a Z\assets\train_u6lujuX_CVtuZ9i.csv')
data


# In[8]:


# Faites une copie de l’objet “data” dans un nouvel objet appelé “df”.
df = data.copy()
print(df)


# In[9]:


# pour afficher toute la base de données
pd.set_option('display.max_rows',df.shape[0]+1)


# In[10]:


df


# In[11]:


#revenir a afficher max 10lignes
pd.set_option('display.max_rows', 10)


# In[12]:


df


# In[13]:


#voir les valeurs manquante
df.info()


# In[14]:


#combien on a de valeur manquante pour chaque colonne
df.isnull().sum().sort_values(ascending=False)


# In[15]:


#valeur qui sont anormale
df.describe()


# In[16]:


#valeur afficher les valeurs de type categorique
df.describe(include='O')


# In[17]:


#le type de chaque valeur
df.dtypes


# In[18]:


# Renseigner les valeur manquantes
            #d'abord afficher les valeur categorique et numerique
cat_data = []
num_data = []
for i,c in enumerate(df.dtypes):
    if c==object:
        cat_data.append(df.iloc[:,i])
    else:
        num_data.append(df.iloc[:,i])
cat_data = pd.DataFrame(cat_data).transpose() #transformer la liste en base de donnée
num_data = pd.DataFrame(num_data).transpose()
#cat_data
#cat_dat


# In[19]:


# pour les variables categoriques on va remplacer les valeur manquantes par valeur qui se repete le plus 
cat_data=cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))  #index 0 pour remplacer par celui qui se repete le plus
cat_data.isnull().sum().any()


# In[20]:


#ex permet de compter le nombre valeur ds une colonne et classer ses valeurs
cat_data['Education'].value_counts()


# In[21]:


# pour les variables numeriques on va remplacer les valeur manquantes par valeur precedente de la meme colonne
num_data.fillna(method='bfill', inplace=True)  #index 0 pour remplacer par celui qui se repete le plus
num_data.isnull().sum().any()


# In[22]:


num_data


# In[23]:


# transformer la colonne target en 0 et 1
target_value={'Y':1, 'N':0}
target=cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)
target=target.map(target_value)
target


# In[24]:


# Remplacer les valeurs categoriques par des valeur numeriques 0,1,2...
le = LabelEncoder()
for i in cat_data:
    cat_data[i]=le.fit_transform(cat_data[i])
cat_data


# In[25]:


# supprimer une colonne: Loan_ID
cat_data.drop('Loan_ID', axis=1, inplace=True)


# In[26]:


#concatener cat_data et num_data(variable categorielle et numerique ) et specifier la colonne target
X=pd.concat([cat_data, num_data], axis=1) #concatener ds une meme base de donnés
y=target                                  #variable cible(independant) on la specifier plus haut target


# In[27]:


X


# In[28]:


y


# In[29]:


#2 Realisation d'une Analyse Exploratoie(EDA)

 # va commencer par la variable target
target.value_counts()


# In[30]:


# la base de données utilisée pour EDA
df=pd.concat([cat_data,num_data,target],axis=1)


# In[31]:


#methode de visualisition
plt.figure(figsize=(8,6))
sns.countplot(target)
yes=target.value_counts()[0]/len(target)
no=target.value_counts()[1]/len(target)
print(f'le pourcentage des credits accordés est: {yes}')
print(f'le pourcentage des credits non accordés est: {no}')


# In[32]:


# Credit history
grid=sns.FacetGrid(df, col='Loan_Status', size=3.2,aspect=1.6)
grid.map(sns.countplot,'Credit_History')


# In[33]:


#sexe
grid=sns.FacetGrid(df, col='Loan_Status', size=3.2,aspect=1.6)
grid.map(sns.countplot,'Gender')


# In[34]:


#mariage
grid=sns.FacetGrid(df, col='Loan_Status', size=3.2,aspect=1.6)
grid.map(sns.countplot,'Married')


# In[35]:


#Education
grid=sns.FacetGrid(df, col='Loan_Status', size=3.2,aspect=1.6)
grid.map(sns.countplot,'Education')


# In[36]:


#revenu du demandeur
plt.scatter(df['ApplicantIncome'],df['Loan_Status'])


# In[37]:


#revenu du demandeur
plt.scatter(df['CoapplicantIncome'],df['Loan_Status'])


# In[38]:


df.groupby('Loan_Status').median()


# In[39]:


# 3 realisation du modele
# Diviser la base de données en une base de données test et d'entrainement
sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train,test in sss.split(X,y):
    X_train,X_test=X.iloc[train],X.iloc[test]
    y_train,y_test=y.iloc[train],y.iloc[test]

print('X_train taille: ', X_train.shape)
print('X_test taille: ', X_test.shape)
print('y_train taille: ', y_train.shape)
print('y_test taille: ', y_test.shape)


# In[40]:


# On va appliquer trois algorithmes Logistic, KNN, DecisionTree

models={
    'LogisticRegression':LogisticRegression(random_state=42),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'DecisionTreeClassifier':DecisionTreeClassifier(max_depth=1,random_state=42)
}

# la fonction de precision
def accu(y_true,y_pred,retu=False):
    acc=accuracy_score(y_true,y_pred)
    if retu:
        return acc
    else:
        print(f'la precision du modèle est: {acc}')

# c'est la fonction d'application des modèle
def train_test_eval(models,X_train,y_train,X_test,y_test):
    for name,model in models.items():
        print(name,':')
        model.fit(X_train,y_train)
        accu(y_test,model.predict(X_test))
        print('-'*30)
        
train_test_eval(models,X_train,y_train,X_test,y_test)


# In[41]:


# creer une nouvel bd
X_2=X[['Credit_History', 'Married', 'CoapplicantIncome' ]]


# In[42]:


# Diviser la base de données en une base de données test et d'entrainement
sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train,test in sss.split(X_2,y):
    X_train,X_test=X_2.iloc[train],X_2.iloc[test]
    y_train,y_test=y.iloc[train],y.iloc[test]

print('X_train taille: ', X_train.shape)
print('X_test taille: ', X_test.shape)
print('y_train taille: ', y_train.shape)
print('y_test taille: ', y_test.shape)


# In[43]:


train_test_eval(models,X_train,y_train,X_test,y_test)


# In[44]:


# Appliquer la regression logistique sur notre base de donnée
Classifier=LogisticRegression() 
Classifier.fit(X_2,y)


# In[45]:


# Enregistrer le modele
pickle.dump(Classifier,open('model.pkl','wb'))


# In[ ]:




