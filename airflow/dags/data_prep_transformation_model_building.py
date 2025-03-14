from datetime import datetime
import pandas as pd
import great_expectations as ge
import matplotlib.pyplot as plt
import seaborn as sns

from airflow import DAG, settings
from airflow.decorators import task
from airflow.operators.bash import BashOperator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# # import sys
# # sys.path.append('/opt/airflow/')
# from .data_preprocessing import pre_processing
# from .churn_prediction import model_prediction

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

@task
def data_pre_processing():
    df = pd.read_csv(DATA_PATH)
    print("Initial Data Snapshot:")
    print(df.head())
    print("\nData Information:")
    print(df.info())

    # ---------------------------
    # 2. Data Cleaning
    # ---------------------------
    # Drop duplicate rows if any.
    df.drop_duplicates(inplace=True)

    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Fill missing values: For object type, fill with mode; for numeric, fill with median.
    for col in df.columns:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
        else:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Special handling for 'TotalCharges' column: Convert to numeric and fill missing values.
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # ---------------------------
    # 3. Data Transformation
    # ---------------------------
    # Convert categorical variables into numerical ones using LabelEncoder.
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # ---------------------------
    # 4. Exploratory Data Analysis (EDA) Visualizations
    # ---------------------------
    # Plot and save the distribution of the target variable, if available.
    if 'Churn' in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x='Churn', data=df)
        plt.title("Churn Distribution")
        plt.savefig('churn_distribution.png')
        plt.close()

    # Plot the correlation heatmap.
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='viridis')
    plt.title("Correlation Heatmap")
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # ---------------------------
    # 5. Feature Scaling
    # ---------------------------
    # Scale numerical features using StandardScaler.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # ---------------------------
    # 6. Save the Preprocessed Data
    # ---------------------------
    df.to_csv('/opt/data/cleaned_customer_churn.csv', index=False)
    # Get Airflow's default engine
    engine = settings.engine
    # Save DataFrame to Airflow's default DB
    df.to_sql('cleaned_data', engine, if_exists='replace', index=False)

    query = "SELECT * FROM cleaned_data"
    
    # Fetch data into DataFrame
    df = pd.read_sql(query, con=engine)
    print(df.info())

    print("Data preprocessing completed and saved to 'cleaned_customer_churn.csv'")
    return "Data preprocessing completed and saved to 'cleaned_customer_churn.csv'"

@task
def model_prediction():
    #df = pd.read_csv('/opt/data/cleaned_customer_churn.csv')
    engine = settings.engine
    print(engine)

    # Example query (fetch all records from 'my_table')
    query = "SELECT * FROM my_table"
    
    # Fetch data into DataFrame
    df = pd.read_sql(query, con=engine)
    print(df.info())
    # ---------------------------
    # 2. Define Features and Target Variable
    # ---------------------------
    # Assuming 'Churn' is the target variable.    
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # ---------------------------
    # 3. Split the Data
    # ---------------------------
    # Use a 70:30 train-test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # ---------------------------
    # 4. Random Forest Model
    # ---------------------------
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    print("=== Random Forest Model Evaluation ===")
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print("Accuracy:", rf_accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_predictions))
    print("Classification Report:")
    print(classification_report(y_test, rf_predictions))

    # Visualize confusion matrix for Random Forest
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest Confusion Matrix")
    plt.savefig('rf_confusion_matrix.png')
    plt.close()

    # ---------------------------
    # 5. Logistic Regression Model
    # ---------------------------
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    print("\n=== Logistic Regression Model Evaluation ===")
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print("Accuracy:", lr_accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_predictions))
    print("Classification Report:")
    print(classification_report(y_test, lr_predictions))

    # Visualize confusion matrix for Logistic Regression
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, lr_predictions), annot=True, fmt='d', cmap='Greens')
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig('lr_confusion_matrix.png')
    plt.close()

    # ---------------------------
    # 6. Additional Model Evaluation Metrics
    # ---------------------------
    # If needed, one could also add ROC curve plotting or other metrics here as in the notebook.
    from sklearn.metrics import roc_curve, auc

    # For Random Forest
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(6,4))
    plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend(loc='lower right')
    plt.savefig('rf_roc_curve.png')
    plt.close()

    # For Logistic Regression
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure(figsize=(6,4))
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc='lower right')
    plt.savefig('lr_roc_curve.png')
    plt.close()

    print("Model evaluation completed. All plots have been saved.")
    return "Model evaluation completed. All plots have been saved."


with DAG(
    dag_id="data_prep_transformation_model_building",
    schedule=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["Data Preparation Transformation Model building"],
    description= "Data Preparation, transformation and model building"
):
    data_validation()  >> data_pre_processing()  >> model_prediction()