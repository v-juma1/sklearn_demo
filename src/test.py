import sys
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import dtreeviz

random_state = 1234  # get reproducible trees
dataset_url = "scikit-learn/data/titanic/train.csv"
dataset = pd.read_csv(dataset_url)
# Fill missing values for Age
dataset.fillna({"Age": dataset.Age.mean()}, inplace=True)
# Encode categorical variables
dataset["Sex_label"] = dataset.Sex.astype("category").cat.codes
dataset["Cabin_label"] = dataset.Cabin.astype("category").cat.codes
dataset["Embarked_label"] = dataset.Embarked.astype("category").cat.codes
features = [
    "Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"
]
target = "Survived"

tree_classifier = DecisionTreeClassifier(max_depth=3,
                                         random_state=random_state)
tree_classifier.fit(dataset[features].values, dataset[target].values)
viz_model = dtreeviz.model(tree_classifier,
                           X_train=dataset[features],
                           y_train=dataset[target],
                           feature_names=features,
                           target_name=target,
                           class_names=["perish", "survive"])
viz_model.view(scale=0.8).save("scikit-learn/log/tree_iris.svg")
