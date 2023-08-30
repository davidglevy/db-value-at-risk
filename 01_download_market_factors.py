# Databricks notebook source
# MAGIC %md
# MAGIC ## Download market factors
# MAGIC We assume that our various assets can be better described by market indicators and various indices, such as the S&P500, crude oil, treasury, or the dow. These indicators will be used later to create input features for our risk models

# COMMAND ----------

# MAGIC %pip install yfinance==0.1.70 dbl-tempo==0.1.17

# COMMAND ----------

import pandas as pd
import yfinance as yf
from var_support.var_helper import getTaskInfo
import datetime as dt
from pyspark.sql.functions import array, col

# COMMAND ----------

config = getTaskInfo(dbutils)

startdate = dt.datetime.strptime(config['yfinance_start'], "%Y-%m-%d").date()
enddate = dt.datetime.strptime(config['yfinance_stop'], "%Y-%m-%d").date()

# COMMAND ----------

factors = {
  '^GSPC':'SP500',
  '^NYA':'NYSE',
  '^XOI':'OIL',
  '^TNX':'TREASURY',
  '^DJI':'DOWJONES'
}

# Create a pandas dataframe where each column contain close index
factors_pdf = pd.DataFrame()
for tick in factors.keys():    
    msft = yf.Ticker(tick)
    raw = msft.history(start=startdate, end=enddate)
    # fill in missing business days
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq='B')
    # use last observation carried forward for missing value
    pdf = raw.reindex(idx, method='pad')
    factors_pdf[factors[tick]] = pdf['Close'].copy()
        
# Pandas does not keep index (date) when converted into spark dataframe
factors_pdf['date'] = idx

factors_df = spark.createDataFrame(factors_pdf)


# COMMAND ----------


market_table = config['market_table']
print(f"About to write our market information to the table [{market_table}]")

(factors_df
    .write
    .mode("overwrite")
    .saveAsTable(market_table)
)
display(spark.table(market_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute market volatility
# MAGIC As mentioned in the introduction, the whole concept of parametric VaR is to learn from past volatility. Instead of processing each day against its closest history sequentially, we can apply a simple window function to compute last X days' worth of market volatility at every single point in time, learning statistics behind those multi variate distributions

# COMMAND ----------

import numpy as np

def get_market_returns():
  
    f_ret_pdf = spark.table(market_table).orderBy('date').toPandas()

    # add date column as pandas index for sliding window
    f_ret_pdf.index = f_ret_pdf['date']
    f_ret_pdf = f_ret_pdf.drop(columns = ['date'])

    # compute daily log returns
    f_ret_pdf = np.log(f_ret_pdf.shift(1)/f_ret_pdf)

    # add date columns
    f_ret_pdf['date'] = f_ret_pdf.index
    f_ret_pdf = f_ret_pdf.dropna()



    return (
    spark
        .createDataFrame(f_ret_pdf)
        .select(F.array(config['feature_names']).alias('features'), F.col('date'))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of recursively querying our data, we can apply a window function so that each insert of our table is "joined" with last X days worth of observations. We can compute statistics of market volatility for each window using simple UDFs

# COMMAND ----------

from pyspark.sql.functions import udf

@udf('array<double>')
def compute_avg(xs):
    import numpy as np
    mean = np.array(xs).mean(axis=0)
    return mean.tolist()
  
@udf('array<array<double>>')
def compute_cov(xs):
    import pandas as pd
    return pd.DataFrame(xs).cov().values.tolist()

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

days = lambda i: i * 86400 
volatility_window = Window.orderBy(F.col('date').cast('long')).rangeBetween(-days(config['past_volatility']), 0)

volatility_df = (
  get_market_returns()
    .select(
      F.col('date'),
      F.col('features'),
      F.collect_list('features').over(volatility_window).alias('volatility')
    )
    .filter(F.size('volatility') > 1)
    .select(
      F.col('date'),
      F.col('features'),
      compute_avg(F.col('volatility')).alias('vol_avg'),
      compute_cov(F.col('volatility')).alias('vol_cov')
    )
)

# COMMAND ----------

volatility_df.write.mode('overwrite').saveAsTable(config['volatility_table'])

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we now have access to up to date indicators at every single point in time. For each day, we know the average of returns and our covariance matrix. These statistics will be used to generate random market conditions in our next notebook.

# COMMAND ----------

display(spark.read.table(config['volatility_table']))
