import base64
import json

import tenseal as ts
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
label_encoder = LabelEncoder()
model_save_path = "D:/My_Projects/SBH/model"
# Load multiple CSV files from a folder
def load_multiple_csv(folder_path):
    all_dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)
def load():
    # Specify the folder path containing CSV files
    folder_path = "D:/My_Projects/SBH/uploads"  # Update with the correct path
    df = load_multiple_csv(folder_path)

    # Identify the outcome column (last column assumed as target)
    outcome_column = df.columns[-1]  # Automatically detect last column as target


    label_encoders = {}
    # Encode all columns with dtype 'object'

    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
        label_encoders[column] = label_encoder
    if outcome_column not in label_encoders:
        label_encoders[outcome_column] = label_encoder.fit(df[outcome_column])  # Convert categories to numbers
    print(label_encoders)


    joblib.dump(label_encoders, os.path.join(model_save_path, "label_encoders.pkl"))

    # Create encryption context
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return df , outcome_column, context

# Step 1: Encrypt the dataset
def encrypt_data(dataf, context, outcome_col):
    encrypted_data = []
    for _, row in dataf.iterrows():
        encrypted_vector = ts.ckks_vector(context, row.drop(outcome_col).tolist()).serialize()
        encrypted_data.append(base64.b64encode(encrypted_vector).decode('utf-8'))  # Convert to base64 string
    return encrypted_data, dataf[outcome_col].tolist()

def encrypt_data1(df, context, outcome_column):
    encrypted_data = []
    for _, row in df.iterrows():
        encrypted_data.append(ts.ckks_vector(context, row.drop(outcome_column).tolist()))
    return encrypted_data, df[outcome_column].tolist()

def save_encrypted_data(encrypted_data, labels, folder, filename):
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
    filepath = os.path.join(folder, filename)  # Full path
    with open(filepath, "w") as f:
        json.dump({"features": encrypted_data, "labels": labels}, f)  # Save as JSON
    print(f"✅ Encrypted data saved at: {filepath}")



# Encrypt the dataset
# encrypted_features, labels = encrypt_data(df, context, outcome_column)
def load_model(df, outcome_column, context):
    # Split dataset into training and testing sets (80-20 split)
    X = df.drop(columns=[outcome_column]).values
    y = df[outcome_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Scale test data with same scaler

    # Train logistic regression model
    model = LogisticRegression(solver='saga', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # **Compute Accuracy**
    y_pred = model.predict(X_test_scaled)
    print("Y predict ---:",y_pred)
    accuracy = accuracy_score(y_test, y_pred)  # Accuracy calculation

    # Encrypt model coefficients
    encrypted_weights = ts.ckks_vector(context, model.coef_[0])
    encrypted_bias = ts.ckks_vector(context, [model.intercept_[0]])

    # Save model
    model_save_path = "D:/My_Projects/SBH/model"
    joblib.dump(model, os.path.join(model_save_path, "logistic_regression_model.pkl"))

    return encrypted_weights, encrypted_bias, accuracy*100

# Step 2: Perform encrypted inference
def encrypted_predict(encrypted_features, encrypted_weights, encrypted_bias):
    encrypted_predictions = []
    for enc_x in encrypted_features:
        enc_y = enc_x.dot(encrypted_weights) + encrypted_bias
        encrypted_predictions.append(enc_y)
    return encrypted_predictions

# encrypted_predictions = encrypted_predict(encrypted_features, encrypted_weights, encrypted_bias)


# Step 3: Decrypt the predictions
def decode(encrypted_predictions):
    label_encoders = joblib.load("D:/My_Projects/SBH/model/label_encoders.pkl")

    first_label_encoder = next(iter(label_encoders.values()))  # Get any encoder
    num_classes = len(first_label_encoder.classes_) if first_label_encoder else 2

    decrypted_predictions = [int(p) for p in [pred.decrypt()[0] for pred in encrypted_predictions]]
    decrypted_predictions = np.clip(decrypted_predictions, 0, num_classes - 1)

    decoded_labels = []
    for p in decrypted_predictions:
        try:
            decoded_labels.append(first_label_encoder.inverse_transform([p])[0])
        except ValueError:
            decoded_labels.append("Unknown")
    print("decode labels:",decoded_labels)
    return decoded_labels


# print("Decrypted Predictions:", decoded_predictions)


# # Load the saved model and test with new unseen data
# def load_and_test_model(model_folder_path, test_folder_path):
#     model = joblib.load(os.path.join(model_folder_path, "logistic_regression_model.pkl"))
#     label_encoder = joblib.load(os.path.join(model_folder_path, "label_encoder.pkl"))
#     test_df = load_multiple_csv(test_folder_path)
#     test_features = test_df.values  # Assuming no target column in test set
#     predictions = model.predict(test_features)
#     decoded_predictions = label_encoder.inverse_transform(predictions)
#     print("Test Predictions:", decoded_predictions)
#
# # Example usage
# model_folder_path = "D:/My_Projects/SBH/model"  # Update with the correct model path
# test_folder_path = "D:/My_Projects/SBH/test"  # Update with the correct test dataset path
# # load_and_test_model(model_folder_path, test_folder_path)