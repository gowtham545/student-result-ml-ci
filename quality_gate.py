
import json
import os
import joblib

# Check that training generated all required files
required_files = [
    "student_result_model.pkl",
    "metrics.json",
    "student_results.csv"
]

for filename in required_files:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Required file missing: {filename}")

# Load the evaluation metrics
with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

# Quality gate: minimum acceptable accuracy is 70%
if accuracy < 0.85:
    raise ValueError(
        f"Quality gate failed: accuracy is {accuracy}"
    )

# Verify that the trained model can be loaded
model = joblib.load("student_result_model.pkl")

print("Quality gate passed!")
print(f"Model accuracy: {accuracy}")
print("All required output files are present.")
print("The trained model can be loaded successfully.")
