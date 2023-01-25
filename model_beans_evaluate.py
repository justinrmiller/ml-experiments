from datasets import load_dataset
from evaluate import evaluator

import json

model_name = "vit-base-beans"

task_evaluator = evaluator("image-classification")
data = load_dataset("beans", split="test[:40]")
results = task_evaluator.compute(
    model_or_pipeline=f"./{model_name}",
    data=data,
    label_column="labels",
    metric="accuracy",
    label_mapping={'angular_leaf_spot': 0, 'bean_rust': 1, 'healthy': 2},
    strategy="bootstrap"
)
print(json.dumps(results, indent=2))
