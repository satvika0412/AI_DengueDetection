import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset (Ensure your CSV has these column names)
# df = pd.read_csv('dengue_data.csv') 

# Sample Data Structure for training:
# WBC, Platelets, Hematocrit, Hemoglobin, Result(1 for Dengue, 0 for Healthy)
data = {
    'WBC': [4500, 2800, 8000, 3100, 5000, 2900],
    'Platelets': [150000, 80000, 250000, 90000, 200000, 75000],
    'Hematocrit': [42, 50, 40, 48, 41, 52],
    'Hemoglobin': [14, 16, 13, 15, 14, 17],
    'Result': [0, 1, 0, 1, 0, 1] 
}
df = pd.DataFrame(data)

X = df[['WBC', 'Platelets', 'Hematocrit', 'Hemoglobin']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
with open('models/dengue_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")