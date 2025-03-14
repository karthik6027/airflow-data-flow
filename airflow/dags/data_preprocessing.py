# preprocessing.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Data Loading and Initial Inspection
# ---------------------------
# Load the dataset – ensure the file 'customer_churn.csv' is in the working directory.

def pre_processing():
    df = pd.read_csv('data//ingested_data//telco-customer-churn-mul-sources.csv')
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
    df.to_csv('data//preprocessed_data//cleaned_customer_churn.csv', index=False)
    print("Data preprocessing completed and saved to 'cleaned_customer_churn.csv'")
    return "Data preprocessing completed and saved to 'cleaned_customer_churn.csv'"