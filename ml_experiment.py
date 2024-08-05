import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Load the Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Infer the model signature
signature = infer_signature(X_train, model.predict(X_train))

# Log experiment details with MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 2000)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train.iloc[:5])

print("Logged data to MLflow with accuracy:", accuracy)
