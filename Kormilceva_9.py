#!/usr/bin/env python
# coding: utf-8

# # Развертывание веб-приложения машинного обучения 

# In[86]:


### УСТАНОВКА
##pip install streamlit


# In[87]:


import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pylab as plt
import seaborn as sns


# In[88]:


st.sidebar.header('User Input Parameters')


# In[89]:


def user_input_features():
    #TYPE = st.multiselect('TYPE', ('НКЛ','ВКЛ','К','Прав.треб.'))
    TYPE = st.sidebar.selectbox('TYPE', ('НКЛ','ВКЛ','К','Прав.треб.'))
    SEGMENT = st.sidebar.selectbox('SEGMENT', ('Micro','MB','CB','KB.'))
    AGE = st.sidebar.slider('AGE', 0.0, 5.0, 35.0)
    EMPLOYEES = st.sidebar.slider('EMPLOYEES', 0.0, 10.0, 10000.0)
    LIQUIDITY = st.sidebar.slider('LIQUIDITY', -10.0, 1.0, 10.0)
    EBIT_INTEREST = st.sidebar.slider('EBIT_INTEREST', -10.0, 10.0, 10.0)
    DEBT_EBIT = st.sidebar.slider('DEBT_EBIT', -10.0, 10.0, 10.0)
    DEBT_REVENUE = st.sidebar.slider('DEBT_REVENUE', -10.0, 10.0, 10.0)
    CREDIT_REVENUE = st.sidebar.slider('CREDIT_REVENUE', -10.0, 10.0, 10.0)
    STABILITY = st.sidebar.slider('STABILITY', -100.0, 10.0, 100.0)

    data = {'AGE': AGE,
            'EMPLOYEES': EMPLOYEES,
            'LIQUIDITY': LIQUIDITY,
            'EBIT_INTEREST': EBIT_INTEREST,
            'DEBT_EBIT': DEBT_EBIT,
            'DEBT_REVENUE': DEBT_REVENUE,
            'CREDIT_REVENUE': CREDIT_REVENUE,
            'STABILITY': STABILITY         
            }
    features = pd.DataFrame(data, index=[0])
    return features


# In[90]:


df_user = user_input_features()


# #Здесь датафрейм, сформированный функцией user_input_features() записываем в переменную df_user.

# In[91]:


df = pd.read_csv("DATASET_2.csv", nrows=500)
df.drop(['RATING','PURPOSE', 'TYPE', 'SEGMENT', 'EQUIPMENT', 'BRANCH', 'OKATO', 'ASSETS', 'LONGTERMLOAN', 'CURRENTLOAN', 'EQUITY', 'REVENUE', 'INTEREST', 'EBIT', 'PROFIT', 'CREDIT', 'CREDITHISTORY', 'RESTRUCTURING', 'LAWSUITS'], axis=1, inplace=True)


# In[92]:


df_mean = df.mean()
df.fillna(df_mean, inplace=True)


# In[93]:


#df.info()


# In[94]:


Y = df['DEFAULT']
X = df.drop(['DEFAULT'], axis=1)


# In[95]:


#dummies.describe()


# In[96]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=17)
model


# In[97]:


model.fit(X, Y)


# In[ ]:





# In[98]:


prediction = model.predict(df_user)


# 
# Получение сведений о прогностической вероятности.

# In[99]:


prediction_proba = model.predict_proba(df_user)


# # Формирование основной панели

# In[100]:


st.write("""
# Credit Scoring
Приложение прогнозирования платежеспособности Заемщиков, адекватно
оценивающее вероятность погашения кредитного продукта, привлекаемого в
рамках кредитных платформ.
""")


# In[101]:


st.write(df_user)


# In[112]:


st.bar_chart(df_user)


# In[102]:


st.subheader('Предсказание платежеспособности потенциального Заемщика')


# In[107]:


st.write('DEFAULT=0 Платежеспособный', 'DEFAULT=0 Неплатежеспособный')


# In[104]:


st.subheader('Prediction')


# In[105]:


st.write(df.DEFAULT[prediction])


# In[106]:


st.subheader('Prediction Probability')


# Вывод данных о прогностической вероятности.

# In[108]:


st.write(prediction_proba)


# # Запуск веб-приложения

# In[99]:


##streamlit run Kormilceva.py


# In[ ]:





# In[ ]:





# In[ ]:




