import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\user\Downloads\child-mind-institute-problematic-internet-use\child-mind-institute-problematic-internet-use\train.csv')

# Load the test dataset
test_data = pd.read_csv(r'C:\Users\user\Downloads\child-mind-institute-problematic-internet-use\child-mind-institute-problematic-internet-use\test.csv')

# Drop the 'id' column as it is not informative
if 'id' in data.columns:
    data = data.drop(columns=['id'])
if 'id' in test_data.columns:
    test_data = test_data.drop(columns=['id'])

# Handle missing values in the target variable by dropping rows with missing target
data = data.dropna(subset=['sii'])

# Separate features and target variable
target = 'sii'
X = data.drop(columns=[target])
y = data[target]

# Align columns of test data to training data
missing_cols = set(X.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

# Ensure the columns order matches between train and test data
test_data = test_data[X.columns]

# Check if the target column exists in the test data and proceed accordingly
if target in test_data.columns:
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]
    has_test_labels = True
else:
    X_test = test_data
    y_test = None
    has_test_labels = False

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Convert all categorical columns to strings to ensure uniformity
for col in categorical_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessor for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model with reduced complexity to avoid overfitting
model = RandomForestClassifier(random_state=42, max_depth=10, min_samples_leaf=4, n_estimators=50)

# Create the full pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop rows with missing values in the training target
y_train = y_train.dropna()
X_train = X_train.loc[y_train.index]

# Train the model
clf.fit(X_train, y_train)

# Make predictions on validation set
y_val_pred = clf.predict(X_val)

# Evaluate the model on validation set
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

# Make predictions on the test set
y_test_pred = clf.predict(X_test)

# Evaluate the model on the test set if labels are available
if has_test_labels:
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred))
else:
    print("Test predictions (without ground truth labels):")
    print(y_test_pred)

# Perform cross-validation to check for overfitting
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

# Print cross-validation results
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Plot cross-validation scores to visualize potential overfitting
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='--', color='b')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy Score')
plt.title('Cross-Validation Scores for Random Forest Model')
plt.grid(True)
plt.show()
