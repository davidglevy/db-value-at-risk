# Helper Functions for the Value at Risk calculations

# Requires yfinance==0.1.70 dbl-tempo==0.1.17

import warnings
import re
from pathlib import Path
import mlflow
warnings.filterwarnings("ignore")

def getTaskInfo(dbutils):

    # We ensure that all objects created in that notebooks will be registered in a user specific database. 
    useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    username = useremail.split('@')[0]
    user_prefix = re.sub('\W', '_', username)

    # Please replace this cell should you want to store data somewhere else.
    database_name = '{}_var'.format(re.sub('\W', '_', username))
  
    # Similar to database, we will store actual content on a given path
    home_directory = '/FileStore/{}/var'.format(username)
    dbutils.fs.mkdirs(home_directory)

    # Where we might stored temporary data on local disk
    temp_directory = f"/tmp/{user_prefix}/var"

    # Where we'll download raw files to
    volume = "raw"

    catalog = "demo"

    config = {
        "user_email" : useremail,
        "user_prefix" : user_prefix,
        "catalog" : catalog, # Change this to your preferred catalog
        "schema" : f"{user_prefix}_var",
        "volume" : volume,
        "temp_directory" : temp_directory,
        'portfolio_table'           : f'{catalog}.{database_name}.portfolio',
        'stock_table'               : f'{catalog}.{database_name}.stocks',
        'market_table'              : f'{catalog}.{database_name}.instruments',
        'volatility_table'          : f'{catalog}.{database_name}.volatility',
        'monte_carlo_table'         : f'{catalog}.{database_name}.monte_carlo',
        'trials_table'              : f'{catalog}.{database_name}.trials',
        'news_bronze'               : f'{catalog}.{database_name}.ws_news_bronze',
        'news_silver'               : f'{catalog}.{database_name}.ws_news_silver',
        'news_gold'                 : f'{catalog}.{database_name}.ws_news_bronze',
        'model_name'                : 'var_{}'.format(re.sub('\W', '_', username)),
        'feature_names'             : ['SP500', 'NYSE', 'OIL', 'TREASURY', 'DOWJONES'],
        'yfinance_start'            : '2018-05-01',
        'yfinance_stop'             : '2020-05-01',
        'model_training_date'       : '2019-09-01',
        'num_runs'                  : 32000,
        'past_volatility'           : 90,
        'num_executors'             : 20,
    }

    return config

def setup_var(dbutils, spark):
    task_info = getTaskInfo(dbutils)

    catalog = task_info['catalog']
    schema = task_info['schema']
    volume = task_info['volume']
    print(f"Creating database [{catalog}.{schema}]")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog}.{schema}")

    print(f"Creating volume for raw data [{volume}]")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

    temp_directory = task_info['temp_directory']
    print(f"Creating local temporary directory {temp_directory}")
    Path(temp_directory).mkdir(parents=True, exist_ok=True)

def setup_mlflow(dbutils):
    task_info = getTaskInfo(dbutils)
    useremail = task_info['user_email']

    experiment_name = f"/Users/{useremail}/var"
    mlflow.set_experiment(experiment_name) 
