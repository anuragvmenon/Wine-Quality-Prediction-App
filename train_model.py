import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # NEW: Import SMOTE
import os

# --- Data Loading and Preparation ---

# Get the wine data from the CSV file.
# Assuming the CSV is in the same directory as this script.
df = pd.read_csv('WineQT.csv')

# Pick the columns we will use for training.
X = df.drop(['Id', 'quality'], axis=1)
y = df['quality']

# Encode the target variable (y) to a zero-indexed format (e.g., 3,4,5 -> 0,1,2).
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the data into training and testing sets.
X_train, _, y_train_encoded, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Make sure all the numbers are on a similar scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- SMOTE Implementation (Addressing Imbalance) ---
print("Applying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_encoded)

# --- Model Training and Saving ---

# This is our smart model that will learn from the balanced data.
model = XGBClassifier(random_state=42)
# Train the model on the SMOTE-resampled data
model.fit(X_train_smote, y_train_smote)

# Create a directory to store the models if it doesn't exist.
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model, the scaler, AND the encoder to files.
model_path = os.path.join(model_dir, 'wine_quality_model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')
encoder_path = os.path.join(model_dir, 'encoder.pkl')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(encoder, encoder_path)

print("The model, scaler, and encoder have been trained and saved successfully!")
print("NOTE: The model was trained using SMOTE to handle class imbalance.")
print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Encoder saved to: {encoder_path}")