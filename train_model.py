import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200]
}
model = xgb.XGBClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.2f}")

# Evaluate on test set
test_accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save model
best_model.save_model('model.json')

# Save encoders
with open('le_income.pkl', 'wb') as f:
    pickle.dump(le_income, f)
with open('le_payment.pkl', 'wb') as f:
    pickle.dump(le_payment, f)