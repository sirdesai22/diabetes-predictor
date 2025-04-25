from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, 'diabetes.csv')

try:
    data = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print("Error: CSV file not found.")
    exit()

X = data.drop('Outcome', axis=1)
y = data['Outcome']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'diabetes_model.joblib')
print("Trained model saved as 'diabetes_model.joblib'")