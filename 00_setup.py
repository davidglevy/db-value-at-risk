# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/value-at-risk on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/market-risk.

# COMMAND ----------

from var_support.var_helper import setup_var


# COMMAND ----------

# MAGIC %md
# MAGIC # Run Value at Risk Setup
# MAGIC The following command does a few basic setup tasks.

# COMMAND ----------

setup_var(dbutils, spark)
