# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:43:22 2020

@author: workstation
"""


#Loading libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.interactive(False) #interactive mode is off, you have to call plt.show() explicitly to pop up figure window.
pip install plotly_express==0.4.0
import plotly_express as px

import seaborn as sns

from matplotlib.ticker import StrMethodFormatter

from IPython.display import display, HTML


from numpy import median
plt.style.use('fivethirtyeight')
%matplotlib inline
sns.set(style="ticks")
sns.set(rc={'figure.figsize':(15,10)})

#loading Dataset

import os

os.getcwd()
os.chdir(r"E:\PYTHON\PYTHON_PROJECT\tariq_project")
train=pd.read_csv("train_data.csv")

test=pd.read_csv("test_data.csv")

train = train.drop(['case_id'], axis=1)
test = test.drop(['case_id'], axis=1)
train['dataset'] = 'train'
test['dataset'] = 'test'
df = pd.concat([train, test])


print(df)


df.info()

df.dropna(inplace=True)

#checking shape of the dataset
df.shape
df.size
df.columns

#Checking data types of each variable
df.dtypes

df.head()

df.tail()

df["Department"].unique()

df["Severity of Illness"].unique()


#Checking for missing values in dataset
#In the dataset missing values are represented as 'NaN'sign
for col in df.columns:
    if df[col].dtype == object:
         print(col,df[col][df[col] == 'NaN'].count())
         




ax = sns.catplot(x="Ward_Type", y="Admission_Deposit",col='Age', data=df, estimator=median,height=7, aspect=.4,kind='bar')

plt.show()



ax1 = sns.catplot(x="Hospital_region_code", y="Admission_Deposit",col='Age', data=df, estimator=median,height=7, aspect=.4,kind='bar')

plt.show()




ax2 = sns.catplot(x="Bed Grade", y="Admission_Deposit",col='Age', data=df, estimator=median,height=7, aspect=.4,kind='bar')

plt.show()

ax3 = sns.catplot(x="Department", y="Admission_Deposit",col='Type of Admission', data=df, estimator=median,height=7, aspect=.7,kind='bar')

plt.show()


ax4=df.pivot_table(index='Age',columns='Severity of Illness',values='Visitors with Patient', aggfunc='sum').plot(kind='barh')

plt.show()


ax5=df.pivot_table(index='Stay',columns='Age',values='Admission_Deposit', aggfunc='sum').plot(kind='barh')
plt.show()



ax6=df.pivot_table(index='Stay',columns='Severity of Illness',values='Admission_Deposit', aggfunc='sum').plot(kind='barh')
plt.show()


ax7= df.groupby(by=['Age','Department'])['Admission_Deposit'].sum().unstack().plot(kind='bar',stacked=True)
plt.show()


ax8= df.groupby(by=['Age','Department'])['Admission_Deposit'].sum().unstack().reset_index().melt(id_vars='Age')
plt.show()








# Ward type and total Admission Deposit according to Age group

ex = sns.catplot(x="Ward_Type", y="Admission_Deposit",col='Age', data=df, estimator=median,height=7, aspect=.5,kind='bar')

plt.show()





ax11 = sns.catplot(x="Ward_Type", y="Admission_Deposit",col='Age', data=df, estimator=sum,height=7, aspect=.4,kind='bar')

plt.show()





ax33=df.groupby(['Department','Age']).Admission_Deposit.sum().nlargest(10).plot(kind='barh',color = ['r', 'b',
                                                                                                    'darkorange','turquoise','lime','teal',
                                                                                                    'gold','purple','peru','olive'])

plt.show()


df.pivot_table(index='Age',columns='Stay',values='Admission_Deposit', aggfunc='sum').plot(kind='barh')




ax12=df.groupby(['Visitors with Patient','Age']).Admission_Deposit.sum().nlargest(10).plot(kind='barh',
                                                                                           color = ['r', 'b',
                                                                                                    'darkorange','turquoise','lime','teal',
                                                                                                    'gold','purple','peru','olive'])
plt.show()







#####################################################
# Mean, sum

deposit_M=df.groupby('Age')['Admission_Deposit'].mean()

deposit_M

deposit_S= df.groupby('Age')['Admission_Deposit'].sum()

deposit_S

Visitor_M= df.groupby('Age')['Visitors with Patient'].mean()
Visitor_M




Visitor_S= df.groupby('Age')['Visitors with Patient'].sum()


Visitor_S

type(Visitor_S)




#Top 5 Admission deposit based on different factors


max_Admission_Deposit_dep = df.groupby('Department')['Admission_Deposit'].sum()
max_Admission_Deposit_dep


max_Admission_Deposit_W = df.groupby('Ward_Type')['Admission_Deposit'].sum()
max_Admission_Deposit_W



max_Admission_Deposit_R = df.groupby('Hospital_region_code')['Admission_Deposit'].sum()
max_Admission_Deposit_R




max_Admission_Deposit_S= df.groupby('Severity of Illness')['Admission_Deposit'].sum()
max_Admission_Deposit_S



max_Admission_Deposit_Stay= df.groupby('Stay')['Admission_Deposit'].sum()
max_Admission_Deposit_Stay







#Hitmap

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), vmin=-1, cmap='coolwarm', annot=True);
plt.show()

#######################################################################





#Pie Chart
Pie1= df['Severity of Illness'].value_counts().plot(kind = 'pie', autopct='%1.2f%%')
plt.show()




Pie2= df['Type of Admission'].value_counts().plot(kind = 'pie', autopct='%1.2f%%')
plt.show()

Pie3= df['Department'].value_counts().plot(kind = 'pie', autopct='%1.2f%%')
plt.show()


Pie4= df['Hospital_region_code'].value_counts().plot(kind = 'pie', autopct='%1.2f%%')
plt.show()

Pie5= df['Stay'].value_counts().plot(kind = 'pie', autopct='%1.2f%%')
plt.show()

Pie6= df['Available Extra Rooms in Hospital'].value_counts().plot(kind = 'pie', autopct='%1.2f%%')
plt.show()


Pie7= df['Hospital_type_code'].value_counts().plot(kind = 'pie', autopct='%1.2f%%')
plt.show()

Pie8= df['Ward_Facility_Code'].value_counts().plot(kind = 'pie', autopct='%1.2f%%')
plt.show()

Pie9= df['Visitors with Patient'].value_counts().plot(kind = 'bar')
plt.show()



#################################
#Histogram: 
X=df['Admission_Deposit']
X
plt.hist(x, bins = 50)

type_Admission= df["Type of Admission"]
Grade= df['Bed Grade']


# Boxplot
bx=sns.boxplot(X)

plt.show()

########################################
Scatter Plot:
    
sns.jointplot(x=df["Available Extra Rooms in Hospital"], y=df["Admission_Deposit"], kind='scatter', s=200, color='m', edgecolor="skyblue", linewidth=2)
 

       
