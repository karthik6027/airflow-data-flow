from flask import Flask
import pandas as pd
import json

app = Flask(__name__)

@app.route("/")
def data_fetch():
    file_path = "/opt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(file_path)
    df = pd.read_csv(file_path)
    df2= df[round(len(df)/2):]    
    return df2.to_json(orient='records')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)