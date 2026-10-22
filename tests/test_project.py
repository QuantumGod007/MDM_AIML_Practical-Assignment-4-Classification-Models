"""Lightweight structural checks; use python -m unittest discover -s tests -v."""
import unittest

from sklearn.pipeline import Pipeline

from classification_models import load_data, model_specs, split_data


class Practical04Tests(unittest.TestCase):
    def test_dataset_shape_classes_and_nulls(self):
        features, labels = load_data()
        self.assertEqual(features.shape, (569, 30))
        self.assertEqual(labels.value_counts().sort_index().to_dict(), {0: 357, 1: 212})
        self.assertFalse(features.isna().any().any())

    def test_stratified_split(self):
        features, labels = load_data()
        x_train, x_test, y_train, y_test = split_data(features, labels)
        self.assertEqual((len(x_train), len(x_test)), (455, 114))
        self.assertEqual(y_test.value_counts().sort_index().to_dict(), {0: 72, 1: 42})
        self.assertTrue(x_train.index.intersection(x_test.index).empty)

    def test_scaled_estimators_are_pipelines(self):
        models = model_specs()
        self.assertEqual(set(models), {"SVM", "Decision Tree", "KNN", "Naive Bayes"})
        for name in ("SVM", "KNN"):
            self.assertIsInstance(models[name][0], Pipeline)
            self.assertEqual(models[name][0].steps[0][0], "scale")


if __name__ == "__main__":
    unittest.main()