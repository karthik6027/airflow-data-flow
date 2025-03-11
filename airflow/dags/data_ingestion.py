import json
from datetime import datetime
import pandas as pd
import os
import requests

from airflow import DAG
from airflow.decorators import task

@task
def extract_from_csv():
    """
  tt
    """
    # Define file path inside Docker
    file_path = "/opt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # Check if file exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df1 = df[:round(len(df)/2)]

    else:
        print("File not found:", file_path)
    
    return df1.to_json()
   #print(os.getcwd())

@task
def extract_from_api():
    """
  
    """
    df2_list = requests.get('http://flask:5000')
    return df2_list.content.decode("utf-8")


@task
def load(df1, df2):
    """
    
    """
    data1= json.loads(df1)
    data2 = json.loads(df2)
    df2_new = pd.DataFrame(data2)
    df1_new = pd.DataFrame(data1)    
    df_combined = pd.concat([df1_new, df2_new], ignore_index=True)
    print(df_combined.info())
    df_combined.to_csv("/opt/data/telco-customer-churn-mul-sources.csv",index=True)
    
    return "file uploaded"



with DAG(
    dag_id="data_ingestion",
    schedule=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["Data Ingestion"],
    description= "Data Ingestion"
):

    extracted_data_from_csv = extract_from_csv()
    extracted_data_from_api = extract_from_api()
    load(extracted_data_from_csv, extracted_data_from_api)
    