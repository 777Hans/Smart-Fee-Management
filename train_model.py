import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv('student_data.csv')

# Feature engineering: Create interaction terms
df['low_income_missed'] = ((df['income_level'] == 'low') & (df['payment_history'] == 'missed')).astype(int)
df['low_income_low_attendance'] = ((df['income_level'] == 'low') & (df['attendance_rate'] < 0.55)).astype(int)
df['missed_low_attendance'] = ((df['payment_history'] == 'missed') & (df['attendance_rate'] < 0.55)).astype(int)

# Define features and target
X = df[['income_level', 'attendance_rate', 'academic_score', 'payment_history', 'low_income_missed', 'low_income_low_attendance', 'missed_low_attendance']]
y = df['fee_default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing
categorical_features = ['income_level', 'payment_history']
numeric_features = ['attendance_rate', 'academic_score', 'low_income_missed', 'low_income_low_attendance', 'missed_low_attendance']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# Pipeline with preprocessor and classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=1.0,  # Balanced weight
        early_stopping_rounds=10
    ))
])

# Hyperparameter distributions (focused)
param_dist = {
    'classifier__n_estimators': randint(100, 250),
    'classifier__learning_rate': uniform(0.01, 0.09),  # [0.01, 0.1]
    'classifier__max_depth': randint(3, 10),
    'classifier__min_child_weight': randint(1, 5),
    'classifier__subsample': uniform(0.8, 0.2),  # [0.8, 1.0]
    'classifier__colsample_bytree': uniform(0.8, 0.2)  # [0.8, 1.0]
}

# Randomized search
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=30,  # Reduced iterations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    error_score='raise'
)

# Preprocess test set for eval_set
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Fit on training data with early stopping
random_search.fit(
    X_train,
    y_train,
    classifier__eval_set=[(X_test_preprocessed, y_test)],
    classifier__verbose=False
)

# Best model
best_model = random_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_names = (random_search.best_estimator_.named_steps['preprocessor']
                 .get_feature_names_out())
importances = random_search.best_estimator_.named_steps['classifier'].feature_importances_
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False).round(4))

# Save full pipeline
joblib.dump(best_model, 'model.pkl')
print("\nModel saved as 'model.pkl'")

# Print best parameters
print("\nBest Parameters:")
print(random_search.best_params_)