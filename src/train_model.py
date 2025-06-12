import pandas as pd
import joblib
import os
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def trainer(ROOT):

    processed_data_path = os.path.join(ROOT, "data", "processed", "cleaned_data.csv")
    model_output_path = os.path.join(ROOT, "models")
    report_path = os.path.join(ROOT, "reports")

    # Create directories if they don't exist
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    def load_data():
        df = pd.read_csv(processed_data_path)
        return df

    def encode_categorical(df):
        encoders = {}
        for col in ['Category', 'Accident_type']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        return df, encoders

    def train_model(df, encoders):
        X = df[['Category', 'Accident_type', 'Year', 'Month']]
        y = df['Value']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LGBMRegressor(objective='regression', min_data_in_leaf=5, random_state=42, verbose = -1) 
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        # Save model & encoder
        joblib.dump(model, os.path.join(model_output_path, "lgbm_model.joblib"))
        joblib.dump(encoders, os.path.join(model_output_path, "label_encoders.joblib"))

        # Save evaluation
        with open(os.path.join(report_path, "model_metrics.txt"), "w") as f:
            f.write(f"MAE: {mae:.2f}\n")
            f.write(f"RMSE: {rmse:.2f}\n")
        print(f"[✔] Evaluation metrics saved to: {report_path}")

    df = load_data()
    df_encoded, encoders = encode_categorical(df)
    train_model(df_encoded, encoders)
