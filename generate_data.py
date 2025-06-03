import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic student data
n_students = 1000
data = {
    'student_id': range(1, n_students + 1),
    'income_level': np.random.choice(['low', 'medium', 'high'], size=n_students, p=[0.4, 0.4, 0.2]),
    'attendance_rate': np.random.uniform(0.5, 1.0, n_students),
    'academic_score': np.random.uniform(50, 100, n_students),
    'payment_history': np.random.choice(['on_time', 'late', 'missed'], size=n_students, p=[0.7, 0.2, 0.1]),
    'fee_default': np.random.choice([0, 1], size=n_students, p=[0.8, 0.2])  # 0: no default, 1: default
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('student_data.csv', index=False)
print("Synthetic data generated and saved as 'student_data.csv'")