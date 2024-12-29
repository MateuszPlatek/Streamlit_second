from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('DataFrame') \
    .master('local[*]') \
    .getOrCreate()
    
from sklearn.datasets import load_iris
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel

log_reg = LogisticRegressionModel.load('saved_log_reg_model')


# More general:
from pyspark.ml.pipeline import PipelineModel
persistedModel = PipelineModel.load('saved_log_reg_model')
