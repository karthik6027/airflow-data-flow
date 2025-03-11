from datetime import datetime
import pandas as pd
import great_expectations as ge



from airflow import DAG
from airflow.decorators import task

# File Paths
DATA_PATH = '/opt/data/telco-customer-churn-mul-sources.csv'
REPORT_PATH = '/opt/data/report.csv'


@task
def data_validation():
    """
    
    """
    df = pd.read_csv(DATA_PATH)
    
    # Convert to Great Expectations dataframe
    ge_df = ge.from_pandas(df)
    
    
    # # Define validation expectations
    expectations = [
        ge_df.expect_column_to_exist("customerID"),
        ge_df.expect_column_values_to_not_be_null("customerID"),
        
        ge_df.expect_column_to_exist("Churn"),
        ge_df.expect_column_values_to_be_in_set("Churn", ["Yes", "No"]),
        
        ge_df.expect_column_to_exist("MonthlyCharges"),
        ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200),
        
        ge_df.expect_column_to_exist("TotalCharges"),
        ge_df.expect_column_values_to_be_between("TotalCharges", min_value='0', max_value='1000'),  # No negative values
        
        ge_df.expect_column_to_exist("tenure"),
        ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=100),
        
        ge_df.expect_column_to_exist("PaymentMethod"),
        ge_df.expect_column_values_to_be_in_set("PaymentMethod", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ]),

        ge_df.expect_column_to_exist("gender"),
        ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"]),

        ge_df.expect_column_to_exist("Contract"),
        ge_df.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"]),
    ]


    # # Collect validation results
    validation_results = {"check": [], "status": []}
    for exp in expectations:
        validation_results["check"].append(exp["expectation_config"]["expectation_type"])
        validation_results["status"].append("Success" if exp["success"] else "Failed")

    # # Save report
    report_df = pd.DataFrame(validation_results)
    report_df.to_csv(REPORT_PATH, index=False)

    print("Data Validation Completed. Report saved at:")



with DAG(
    dag_id="data_prep_transformation",
    schedule=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["Data Preparation and transformation"],
    description= "Data Preparation and transformation"
):
    data_validation()