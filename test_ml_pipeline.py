
import json
import os
import unittest
import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(
            os.path.exists("student_result_model.pkl")
        )

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_dataset_file_exists(self):
        self.assertTrue(os.path.exists("student_results.csv"))

    def test_metrics_are_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)
        self.assertGreater(metrics["training_records"], 0)
        self.assertGreater(metrics["testing_records"], 0)

    def test_model_can_be_loaded(self):
        model = joblib.load("student_result_model.pkl")
        self.assertIsNotNone(model)

    def test_dataset_has_required_columns(self):
        data = pd.read_csv("student_results.csv")

        expected_columns = {
            "study_hours",
            "attendance",
            "previous_marks",
            "result"
        }

        self.assertTrue(
            expected_columns.issubset(set(data.columns))
        )


if __name__ == "__main__":
    unittest.main()
