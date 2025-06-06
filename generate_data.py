import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, chi2_contingency

# Set random seed for reproducibility
np.random.seed(42)

# Generate data
n_samples = 1000
income_levels = ['low', 'medium', 'high']
payment_histories = ['on_time', 'late', 'missed']

# Initialize data with realistic distributions
data = {
    'student_id': range(1, n_samples + 1),
    'income_level': np.random.choice(income_levels, n_samples, p=[0.4, 0.4, 0.2]),
    'payment_history': np.random.choice(payment_histories, n_samples, p=[0.6, 0.3, 0.1])
}

df = pd.DataFrame(data)

# Generate continuous features with bias toward defaults
def generate_attendance(income, payment):
    base = np.random.uniform(0.5, 1.0)
    if income == 'low' or payment in ['late', 'missed']:
        base -= np.random.uniform(0.1, 0.3)
    return min(max(base, 0.5), 1.0)

def generate_score(income, payment):
    base = np.random.uniform(50, 100)
    if income == 'low' or payment in ['late', 'missed']:
        base -= np.random.uniform(10, 30)
    return min(max(base, 50), 100)

df['attendance_rate'] = df.apply(lambda x: generate_attendance(x['income_level'], x['payment_history']), axis=1)
df['academic_score'] = df.apply(lambda x: generate_score(x['income_level'], x['payment_history']), axis=1)

# Generate fee_default with strong correlations
def assign_default(row):
    prob = 0.0
    # Income level impact
    if row['income_level'] == 'low':
        prob += 0.45
    elif row['income_level'] == 'medium':
        prob += 0.20
    elif row['income_level'] == 'high':
        prob -= 0.15
    # Payment history impact
    if row['payment_history'] == 'missed':
        prob += 0.55
    elif row['payment_history'] == 'late':
        prob += 0.35
    elif row['payment_history'] == 'on_time':
        prob -= 0.20
    # Attendance rate impact
    if row['attendance_rate'] < 0.55:
        prob += 0.45
    # Academic score impact
    if row['academic_score'] < 55:
        prob += 0.45
    # Interactions
    if row['income_level'] == 'low' and row['payment_history'] == 'missed':
        prob += 0.40
    if row['income_level'] == 'low' and row['attendance_rate'] < 0.55:
        prob += 0.25
    if row['payment_history'] == 'missed' and row['attendance_rate'] < 0.55:
        prob += 0.20
    # Cap probability
    return np.random.rand() < min(max(prob, 0.0), 0.70)

df['fee_default'] = df.apply(assign_default, axis=1)

# Validate default count
default_count = df['fee_default'].sum()
if not (330 <= default_count <= 370):
    print(f"Warning: Default count {default_count} is outside target range [330, 370].")

# Save dataset
df.to_csv('student_data.csv', index=False)
print("\nGenerated student_data.csv with stronger feature correlations.")
print("Class distribution:")
print(df['fee_default'].value_counts())

# Correlation analysis
print("\nFeature correlations with fee_default:")
for col in ['attendance_rate', 'academic_score']:
    corr, _ = pointbiserialr(df['fee_default'], df[col])
    print(f"{col}: {corr:.4f}")

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.size == 0 or min(confusion_matrix.shape) < 2:
        return 0.0, 1.0
    chi2, p, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    return v, p

for col in ['income_level', 'payment_history']:
    v, p = cramers_v(df[col], df['fee_default'])
    print(f"{col}: Cramer’s V = {v:.4f}, p-value = {p:.2e}")

print("\nPoint-biserial correlations for categorical feature levels:")
for col in ['income_level', 'payment_history']:
    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
    for dummy_col in dummies.columns:
        corr, _ = pointbiserialr(df['fee_default'], dummies[dummy_col])
        print(f"{dummy_col}: {corr:.4f}")

print("\nFeature distributions:")
print(df[['income_level', 'payment_history']].describe(include='object'))
print("\nContinuous features:")
print(df[['attendance_rate', 'academic_score']].describe())

print("\nDefault rates by category:")
for col in ['income_level', 'payment_history']:
    print(f"\n{col}:")
    print(df.groupby(col)['fee_default'].mean().round(4))

print("\nDefault rate for interactions:")
subset = df[(df['income_level'] == 'low') & (df['payment_history'] == 'missed')]
print(f"low income + missed: {subset['fee_default'].mean().round(4)} (n={len(subset)})")
subset = df[(df['income_level'] == 'low') & (df['attendance_rate'] < 0.55)]
print(f"low income + low attendance: {subset['fee_default'].mean().round(4)} (n={len(subset)})")
subset = df[(df['payment_history'] == 'missed') & (df['attendance_rate'] < 0.55)]
print(f"missed + low attendance: {subset['fee_default'].mean().round(4)} (n={len(subset)})")