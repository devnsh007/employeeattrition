from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd


# Load the dataset and to handle wheather there is missing values and encoding categorical variables, and scaling numerical features
dataset = pd.read_csv('IBMEmployee_data.csv')

# Preprocessing steps
categorical_cols = ['Attrition', 'BusinessTravel', 'Department',
                    'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                  'RelationshipSatisfaction',  'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
# Create transformers for the pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))])
# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])
# Scaling step
scaler = StandardScaler()
# Define the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', scaler)])
# Apply the pipeline to the dataset
dataset = pipeline.fit_transform(dataset)
# Convert the preprocessed dataset back to a DataFrame
dataset = pd.DataFrame(dataset, columns=pipeline.get_feature_names_out())

# Print the preprocessed dataset
print(dataset.head())
print(dataset.columns)
# Assuming 'cat__Attrition_Yes' column represents attrition
y = dataset['cat__Attrition_Yes']

# Selecting the features (all columns except the target variable)
X = dataset.drop(columns=['cat__Attrition_Yes', 'cat__Attrition_No'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Binarize the target variable(because y contains values for 0.5 or more )
y = np.where(y >= 0.5, 1, 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Precision:", precision_score(y_test, y_pred_logreg, average='binary'))
print("Recall:", recall_score(y_test, y_pred_logreg, average='binary'))
print("F1-score:", f1_score(y_test, y_pred_logreg, average='binary'))
print(classification_report(y_test, y_pred_logreg))
# Random Forest
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_scaled, y_train)

y_pred_rfc = rfc.predict(X_test_scaled)

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rfc))
print("Precision:", precision_score(y_test, y_pred_rfc, average='binary'))
print("Recall:", recall_score(y_test, y_pred_rfc, average='binary'))
print("F1-score:", f1_score(y_test, y_pred_rfc, average='binary'))
print(classification_report(y_test, y_pred_rfc))
# SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)
print("\nSVM:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm, average='binary'))
print("Recall:", recall_score(y_test, y_pred_svm, average='binary'))
print("F1-score:", f1_score(y_test, y_pred_svm, average='binary'))
print(classification_report(y_test, y_pred_svm))
# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Create a new Random Forest model for GridSearchCV
rfc = RandomForestClassifier(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid,
                           cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and the corresponding ROC-AUC score
print("Best parameters:", grid_search.best_params_)
print("Best ROC-AUC score:", grid_search.best_score_)

# Fit the Random Forest model with the best parameters
best_rfc = RandomForestClassifier(**grid_search.best_params_, random_state=42)
best_rfc.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
y_pred_best_rfc = best_rfc.predict(X_test_scaled)
print("Random Forest (best parameters):")
print("Accuracy:", accuracy_score(y_test, y_pred_best_rfc))
print("Precision:", precision_score(y_test, y_pred_best_rfc, average='binary'))
print("Recall:", recall_score(y_test, y_pred_best_rfc, average='binary'))
print("F1-score:", f1_score(y_test, y_pred_best_rfc, average='binary'))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_best_rfc))

# Feature Selection using Recursive Feature Elimination (RFE) with the best RandomForestClassifier model
rfe = RFE(estimator=best_rfc, n_features_to_select=10)
rfe.fit(X_train_scaled, y_train)
X_train_selected = rfe.transform(X_train_scaled)
X_test_selected = rfe.transform(X_test_scaled)


# Model Ensemble: AdaBoostClassifier
ada_boost = AdaBoostClassifier(
    base_estimator=rfc, n_estimators=100, random_state=42)
ada_boost.fit(X_train_selected, y_train)
y_pred_ada_boost = ada_boost.predict(X_test_selected)

# Model Ensemble: GradientBoostingClassifier
gradient_boost = GradientBoostingClassifier(n_estimators=100, random_state=42)
gradient_boost.fit(X_train_selected, y_train)
y_pred_gradient_boost = gradient_boost.predict(X_test_selected)

# Evaluate models
print("\nEvaluation Results:")
print("\nAdaBoostClassifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_ada_boost))
print("Precision:", precision_score(y_test, y_pred_ada_boost))
print("Recall:", recall_score(y_test, y_pred_ada_boost))
print("F1-score:", f1_score(y_test, y_pred_ada_boost))
print(classification_report(y_test, y_pred_ada_boost))

print("\nGradientBoostingClassifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_gradient_boost))
print("Precision:", precision_score(y_test, y_pred_gradient_boost))
print("Recall:", recall_score(y_test, y_pred_gradient_boost))
print("F1-score:", f1_score(y_test, y_pred_gradient_boost))
print(classification_report(y_test, y_pred_gradient_boost))
