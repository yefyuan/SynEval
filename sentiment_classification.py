import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

# Load data from CSV files
def load_data(file_path):
    return pd.read_csv(file_path)

# Load real and synthetic data
real_data = load_data('llama2_real.csv')
synthetic_data = load_data('Data/llama2.csv')

# Prepare the data by extracting 'text' and 'rating'
def prepare_data(df):
    df['rating'] = df['rating'].astype(int)  # Convert ratings to integer for classification
    return df[['text', 'rating']]

real_pairs = prepare_data(real_data)
synthetic_pairs = prepare_data(synthetic_data)

# Ensure some initial samples are always in the test set
initial_test = real_pairs.iloc[:50].sample(n=30, random_state=42)
remaining_data = pd.concat([real_pairs.iloc[50:], real_pairs.iloc[:50]]).drop(initial_test.index)

# Split the remaining data
train_real, test_real_remaining = train_test_split(remaining_data, test_size=0.2, random_state=42)

# Combine the manually selected initial tests with the remaining tests
test_real = pd.concat([initial_test, test_real_remaining])

# Feature extraction using TF-IDF with bi-grams
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)

# Fit vectorizer on the training data and transform the training and test data
X_train_real = vectorizer.fit_transform(train_real['text'])
X_train_synthetic = vectorizer.transform(synthetic_pairs['text'])
X_test_real = vectorizer.transform(test_real['text'])

# Random Forest Classifier with hyperparameter tuning
rf_model_real = RandomForestClassifier(random_state=42)
rf_model_synthetic = RandomForestClassifier(random_state=42)

# Grid search to find best parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_real = GridSearchCV(rf_model_real, param_grid, cv=3)
grid_synthetic = GridSearchCV(rf_model_synthetic, param_grid, cv=3)

grid_real.fit(X_train_real, train_real['rating'])
grid_synthetic.fit(X_train_synthetic, synthetic_pairs['rating'])

# Predictions using both models
pred_real = grid_real.predict(X_test_real)
pred_synthetic = grid_synthetic.predict(X_test_real)

# Evaluate the models using Mean Absolute Error (MAE)
mae_real = mean_absolute_error(test_real['rating'], pred_real)
mae_synthetic = mean_absolute_error(test_real['rating'], pred_synthetic)

# Custom accuracy calculation with increased tolerance
def custom_accuracy_with_tolerance(y_true, y_pred, tolerance=1.2):
    """Calculate accuracy where predictions within a certain class difference are correct."""
    return np.mean(np.abs(y_true - y_pred) < tolerance)

accuracy_real = custom_accuracy_with_tolerance(test_real['rating'], pred_real)
accuracy_synthetic = custom_accuracy_with_tolerance(test_real['rating'], pred_synthetic)

print(f"MAE for the real data model: {mae_real}")
print(f"MAE for the synthetic data model: {mae_synthetic}")
print(f"Accuracy for the real data model with increased tolerance: {accuracy_real:.2%}")
print(f"Accuracy for the synthetic data model with increased tolerance: {accuracy_synthetic:.2%}")


