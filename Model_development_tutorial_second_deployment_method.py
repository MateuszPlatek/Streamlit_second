#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql import Row

spark = SparkSession.builder \
    .appName('DataFrame') \
    .master('local[*]') \
    .getOrCreate()


# In[9]:


from sklearn.datasets import load_iris
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression


# In[5]:


iris = load_iris()
X, y = iris.data, iris.target


# In[26]:


features_names = [
 'SepalLengthCm',
 'SepalWidthCm',
 'PetalLengthCm',
 'PetalWidthCm']


# In[31]:


X_pd = pd.DataFrame(X, columns = features_names)
y_pd = pd.DataFrame(y, columns = ["target"])


# In[42]:


df_pd = pd.concat([X_pd, y_pd], axis=1) 


# In[44]:


df0 = spark.createDataFrame(df_pd)


# In[23]:


# y_pd["Target"].unique()


# In[68]:


X_pd.head()


# In[27]:


assembler = VectorAssembler(inputCols = features_names, outputCol = "features")


# In[45]:


df_final = assembler.transform(df0)


# In[ ]:


# df_final = X_train2.withColumn('target', y_train.select('target')['target']) #doesn't work


# In[47]:


log_reg_obj = LogisticRegression(featuresCol = 'features', labelCol = 'target')


# In[48]:


log_reg = log_reg_obj.fit(df_final)


# In[59]:


log_reg.save('saved_log_reg_model')


# In[60]:





# In[65]:


# from pyspark.ml.classification import LogisticRegressionModel


# In[63]:


# mPath = 'saved_log_reg_model'


# In[66]:


# log_reg2 = LogisticRegressionModel.load(mPath)


# In[ ]:




