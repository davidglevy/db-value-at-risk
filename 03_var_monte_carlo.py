# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/value-at-risk on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/market-risk.

# COMMAND ----------

# MAGIC %md
# MAGIC # Monte Carlo
# MAGIC In this notebook, we use our model created in previous stage and run monte carlo simulations in parallel using **Apache Spark**. For each simulated market condition sampled from a multi variate distribution, we will predict our hypothetical instrument returns. By storing all of our data back into **Delta Lake**, we will create a data asset that can be queried on-demand across multiple down stream use cases

# COMMAND ----------

from var_support.var_helper import getTaskInfo
from datetime import timedelta, date, datetime as dt
import datetime
from pyspark.sql.functions import col, lit, udf, struct, collect_list
import pandas as pd
import numpy as np

# COMMAND ----------

config = getTaskInfo(dbutils)

# We will generate monte carlo simulation for every week since we've built our model
today = dt.strptime(config['yfinance_stop'], '%Y-%m-%d')
first = dt.strptime(config['model_training_date'], '%Y-%m-%d')
run_dates = pd.date_range(first, today, freq='w')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Market volatility
# MAGIC As we've pre-computed all statistics at ingest time, we can easily retrieve the most recent statistical distribution of market indicators for each date we want to run monte carlo simulation against. We can access temporal information using asof join of our tempo `library`

# COMMAND ----------

from tempo import *
market_tsdf = TSDF(spark.read.table(config['volatility_table']), ts_col='date')
rdates_tsdf = TSDF(spark.createDataFrame(pd.DataFrame(run_dates, columns=['date'])), ts_col='date')

# COMMAND ----------

volatility_df = rdates_tsdf.asofJoin(market_tsdf).df.select(
  col('date'),
  col('right_vol_cov').alias('vol_cov'),
  col('right_vol_avg').alias('vol_avg')
)

display(volatility_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribute trials
# MAGIC By fixing a seed strategy, we ensure that each trial will be independant (no random number will be the same) as well as enforcing full reproducibility should we need to process the same experiment twice

# COMMAND ----------

# create a dataframe of seeds so that each trial will result in a different simulation
# each executor is responsible for num_instruments * ( total_runs / num_executors ) trials
def create_seed_df():
  runs = config['num_runs']
  seed_df = pd.DataFrame(list(np.arange(0, runs)), columns = ['trial_id'])
  return spark.createDataFrame(seed_df)

# COMMAND ----------

# provided covariance matrix and average of market indicators, we sample from a multivariate distribution
# we allow a seed to be passed for reproducibility
# whilst many data scientists may add a seed as np.random.seed(seed), we have to appreciate the distributed nature 
# of our process and the undesired side effects settings seeds globally
# instead, use rng = np.random.default_rng(seed)

@udf('array<float>')
def simulate_market(vol_avg, vol_cov, seed):
  import numpy as np
  rng = np.random.default_rng(seed)
  return rng.multivariate_normal(vol_avg, vol_cov).tolist()

# COMMAND ----------

market_conditions_df = (
  volatility_df
    .join(create_seed_df())
    .withColumn('features', simulate_market('vol_avg', 'vol_cov', 'trial_id'))
    .select('date', 'features', 'trial_id')
)

# COMMAND ----------

display(market_conditions_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Since this was an expensive operation to cross join each trial ID with each simulated market condition, we can save that table as a delta table that we can process downstream. Furthermore, this table is generic as we only sampled points from known market volatility and did not take investment returns into account. New models and new trading strategies could be executed off the back of the exact same data without having to run this expensive process.

# COMMAND ----------

(
  market_conditions_df
    .repartition(config['num_executors'], 'date')
    .write
    .mode("overwrite")
    .saveAsTable(config['monte_carlo_table'])
)  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute returns
# MAGIC Finally, we can leverage our model created earlier to predict our investment return for each stock given generated market indicators

# COMMAND ----------

import mlflow
model_udf = mlflow.pyfunc.spark_udf(
  model_uri='models:/{}/production'.format(config['model_name']), 
  result_type='float', 
  spark=spark
)

# COMMAND ----------

simulations_df = (
  spark.table(config['monte_carlo_table'])
    .join(spark.table(config['portfolio_table']).select('ticker'))
    .withColumn('return', model_udf(struct('ticker', 'features')))
    .drop('features')
)

display(simulations_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Although we processed our simulated market conditions as a large table made of very few columns, we may want to create a better data asset by wraping all trials into well defined vectors. This asset will help us manipulate vectors through simple aggregated functions using the `Summarizer` class from `pyspark.ml.stat` (see next notebook)

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT

@udf(VectorUDT())
def to_vector(xs, ys):
  return Vectors.sparse(config['num_runs'], zip(xs, ys))

# COMMAND ----------

simulations_vectors_df = (
  simulations_df
    .groupBy('date', 'ticker')
    .agg(
      collect_list('trial_id').alias('xs'),
      collect_list('return').alias('ys')
    )
    .select(
      col('date'),
      col('ticker'),
      to_vector(col('xs'), col('ys')).alias('returns')
    )
)

# COMMAND ----------

(
  simulations_vectors_df
    .write
    .mode("overwrite")
    .saveAsTable(config['trials_table'])
)  

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we make it easy to extract specific slices of our data asset by optimizing our table for faster read.

# COMMAND ----------

spark.sql('OPTIMIZE {} ZORDER BY (`date`, `ticker`)'.format(config['trials_table']))
