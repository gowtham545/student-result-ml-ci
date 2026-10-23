
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a demonstration student dataset
data = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
                    4, 5, 6, 7, 8, 1, 3, 5, 6, 7],
    "attendance": [40, 45, 50, 55, 60, 65, 70, 90, 50, 55,
                   60, 70, 75, 80, 95, 35, 55, 65, 85, 90],
    "previous_marks": [20, 25, 30, 35, 40, 50, 60, 85, 25, 35,
                       45, 55, 65, 75, 90, 15, 35, 60, 70, 80],
    "result": [0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
               0, 1, 1, 1, 1, 0, 0, 1, 1, 1]
})

# Save the generated demonstration dataset
data.to_csv("student_results.csv", index=False)

# Separate input features and target
X = data[["study_hours", "attendance", "previous_marks"]]
y = data["result"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train the ML model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save the trained model
joblib.dump(model, "student_result_model.pkl")

# Save evaluation metrics
metrics = {
    "accuracy": round(float(accuracy), 4),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model training completed successfully!")
print("Accuracy:", metrics["accuracy"])
print("Training records:", metrics["training_records"])
print("Testing records:", metrics["testing_records"])
