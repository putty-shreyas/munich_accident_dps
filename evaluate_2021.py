import os
import pandas as pd
import joblib
import json


def evaluate_pred_2021(ROOT):

    load_model_path = os.path.join(ROOT, "models", "lgbm_model.joblib")
    load_test_data_path = os.path.join(ROOT, "data", "processed", "test_cleaned_data.csv")
    load_ground_truth = os.path.join(ROOT, "reports", "ground_truth_2021_01.json") 
    load_encoder_path = os.path.join(ROOT, "models", "label_encoders.joblib")

    # Load trained model & encoder
    model = joblib.load(load_model_path)
    encoder = joblib.load(load_encoder_path)

    # Load raw test data
    df_test = pd.read_csv(load_test_data_path)

    # Load ground truth JSON
    with open(load_ground_truth, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Verify it's the expected record
    if (
        gt_data["Category"] == "Alkoholunfälle" and
        gt_data["Accident_type"] == "insgesamt" and
        gt_data["Year"] == 2021 and
        gt_data["Month"] == 1
    ):
        gt_entry = gt_data
    else:
        raise ValueError("⚠️ Ground truth record does not match expected fields!")

    # Find matching row in test data for prediction
    X_2021 = df_test[
    (df_test['Category'] == "Alkoholunfälle") &
    (df_test['Accident_type'] == "insgesamt") &
    (df_test['Year'] == 2021) &
    (df_test['Month'] == 1)
    ].drop(columns=["Value"])

     # Apply label encoding
    for col in ['Category', 'Accident_type']:
        le = encoder[col]
        X_2021[col] = le.transform(X_2021[col])

    # Predict
    y_pred = model.predict(X_2021)[0]

    # Get true value from saved json file
    y_true = gt_entry["Value"]

    # Save to reports
    report_df = pd.DataFrame([{
    "year": 2021,
    "month": 1,
    "actual": y_true,
    "prediction": round(y_pred, 2)    
    }])

    report_df.to_csv(os.path.join(ROOT, "reports", "prediction_vs_actual.csv"), index=False)
    print(f"✅ Prediction: {y_pred:.2f}, Ground Truth from JSON: {y_true}")