import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv('student_data.csv')

# Preprocess data
le_income = LabelEncoder()
le_payment = LabelEncoder()
df['income_level'] = le_income.fit_transform(df['income_level'])
df['payment_history'] = le_payment.fit_transform(df['payment_history'])

# Features and target
X = df[['income_level', 'attendance_rate', 'academic_score', 'payment_history']]
y = df['fee_default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('le_income.pkl', 'wb') as f:
    pickle.dump(le_income, f)
with open('le_payment.pkl', 'wb') as f:
    pickle.dump(le_payment, f)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")