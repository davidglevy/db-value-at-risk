# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/value-at-risk on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/market-risk.

# COMMAND ----------

# MAGIC %md
# MAGIC # Create portfolio
# MAGIC In this notebook, we will use `yfinance` to download stock data for 40 equities in an equal weighted hypothetical Latin America portfolio. We show how to use pandas UDFs to better distribute this process efficiently and store all of our output data as a Delta table. 

# COMMAND ----------

# MAGIC %pip install yfinance==0.1.70 dbl-tempo==0.1.17

# COMMAND ----------

from var_support.var_helper import getTaskInfo
import pandas as pd
from io import StringIO
import plotly.graph_objects as go
from pyspark.sql.functions import col, asc, lit, pandas_udf, PandasUDFType
from pyspark.sql.types import *
import datetime as dt
import yfinance as yf


# COMMAND ----------

# MAGIC %md
# MAGIC # Run Value at Risk Setup
# MAGIC The following command does a few basic setup tasks.

# COMMAND ----------

config = getTaskInfo(dbutils)

# COMMAND ----------

portfolio = """
country,company,ticker,industry
CHILE,Banco de Chile,BCH,Banks
CHILE,Banco Santander-Chile,BSAC,Banks
CHILE,Compañía Cervecerías Unidas S.A.,CCU,Beverages
CHILE,Enersis Chile SA Sponsored ADR,ENIC,Electricity
CHILE,"SQM-Sociedad Química y Minera de Chile, S.A.",SQM,Chemicals
COLOMBIA,BanColombia S.A.,CIB,Banks
COLOMBIA,Ecopetrol S.A.,EC,Oil & Gas Producers
COLOMBIA,Grupo Aval Acciones y Valores S.A,AVAL,Financial Services
MEXICO,"América Móvil, S.A.B. de C.V.",AMX,Mobile Telecommunications
MEXICO,CEMEX S.A.B. de C.V. (CEMEX),CX,Construction & Materials
MEXICO,"Coca-Cola FEMSA, S.A.B. de C.V.",KOF,Beverages
MEXICO,"Controladora Vuela Compañía de Aviación, S.A.B. de C.V",VLRS,Travel & Leisure
MEXICO,"Fomento Económico Mexicano, S.A.B. de C.V. (FEMSA)",FMX,Beverages
MEXICO,"Grupo Aeroportuario del Pacífico, S.A.B. de C.V. (GAP)",PAC,Industrial Transportation
MEXICO,"Grupo Aeroportuario del Sureste, S.A. de C.V. (ASUR)",ASR,Industrial Transportation
MEXICO,"Grupo Simec, S.A. De CV. (ADS)",SIM,Industrial Metals & Mining
MEXICO,"Grupo Televisa, S.A.",TV,Media
PANAMA,"Banco Latinoamericano de Comercio Exterior, S.A.",BLX,Banks
PANAMA,"Copa Holdings, S.A.",CPA,Travel & Leisure
PERU,Cementos Pacasmayo S.A.A.,CPAC,Construction & Materials
PERU,Southern Copper Corporation,SCCO,Industrial Metals & Mining
PERU,Fortuna Silver Mines Inc.,FSM,Mining
PERU,Compañía de Minas Buenaventura S.A.,BVN,Mining
PERU,Credicorp Ltd.,BAP,Banks
"""

# create equally weighted index
portfolio_pdf = pd.read_csv(StringIO(portfolio))
portfolio_pdf['weight'] = 1.0
portfolio_pdf['weight'] = portfolio_pdf['weight'] / portfolio_pdf.shape[0]

# Convert to Spark
portfolio_df = spark.createDataFrame(portfolio_pdf)
display(portfolio_df)

# COMMAND ----------

portfolio_df.write.mode('overwrite').saveAsTable(config['portfolio_table'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download stock data
# MAGIC We download stock market data (end of day) from yahoo finance, ensure time series are properly indexed and complete.

# COMMAND ----------

startdate = dt.datetime.strptime(config['yfinance_start'], "%Y-%m-%d").date()
enddate = dt.datetime.strptime(config['yfinance_stop'], "%Y-%m-%d").date()

# COMMAND ----------

schema = StructType(
  [
    StructField('ticker', StringType(), True), 
    StructField('date', TimestampType(), True),
    StructField('open', DoubleType(), True),
    StructField('high', DoubleType(), True),
    StructField('low', DoubleType(), True),
    StructField('close', DoubleType(), True),
    StructField('volume', DoubleType(), True),
  ]
)

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def fetch_tick(group, pdf):
  tick = group[0]
  msft = yf.Ticker(tick)
  raw = msft.history(start=startdate, end=enddate)[['Open', 'High', 'Low', 'Close', 'Volume']]
  # fill in missing business days
  idx = pd.date_range(startdate, enddate, freq='B')
  # use last observation carried forward for missing value
  output_df = raw.reindex(idx, method='pad')
  # Pandas does not keep index (date) when converted into spark dataframe
  output_df['date'] = output_df.index
  output_df['ticker'] = tick    
  output_df = output_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
  return output_df

# COMMAND ----------

portfolio_table = config['portfolio_table']
stock_table = config['stock_table']

print(f"About to create our stock data in table [{stock_table}] from entries in [{portfolio_table}] using yfinance")
portfolio_df = spark.table(portfolio_table)
    
(portfolio_df.groupBy('ticker')
    .apply(fetch_tick)
    .write
    .mode('overwrite')
    .saveAsTable(stock_table))

display(spark.table(stock_table))

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks runtime come prepackaged with many python libraries such as plotly. We can represent a given instrument through a candlestick visualization

# COMMAND ----------

ticker = portfolio_pdf.iloc[0].ticker
print(f"About to show ticker data for {ticker}")

stock_pdf = (
  spark.table(stock_table)
    .filter(col('ticker') == lit(ticker))
    .orderBy(asc('date'))
    .toPandas()
)

layout = go.Layout(
  autosize=False,
  width=1600,
  height=800,
)

fig = go.Figure(
  data=[go.Candlestick(
    x=stock_pdf['date'], 
    open=stock_pdf['open'], 
    high=stock_pdf['high'], 
    low=stock_pdf['low'], 
    close=stock_pdf['close']
  )],
  layout=layout
)

fig.show()
