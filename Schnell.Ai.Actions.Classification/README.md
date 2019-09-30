# schnell.AI Actions (Classification)

This module provides all components required for data-classification.

## Installation
``` powershell
schnell.ai module-i Schnell.Ai.Actions.Classification 0.9.1
```

## Trainer: Schnell.Ai.Actions.Classification.ClassificationTrainer

Use the classification-trainer to train a new model.

### Configuration

| Name         | Type    | Description                                                                      | Default             |
|--------------|---------|----------------------------------------------------------------------------------|---------------------|
| Model        | string* | Name of model to train                                                           |                     |
| TrainingData | string* | Name of dataset to use for training-data                                         |                     |
| UseCaching   | boolean | Use Caching of the data                                                          | false               |
| Algorithm    | string  | Classification-Algorithm (Values: 'SDCA_MaximumEntropy', 'LBFGS_MaximumEntropy') | SDCA_MaximumEntropy |

Properties with a type, ending to '*' are required.

## Tester: Schnell.Ai.Actions.Classification.ClassificationTester

Use the classification-tester to test a model against test-data.

### Configuration

| Name           | Type    | Description                            | Default |
|----------------|---------|----------------------------------------|---------|
| Model          | string* | Name of model to train                 |         |
| TestData       | string* | Name of dataset to use for test-data   |         |
| TestResultData | string* | Name of dataset to use for result-data |         |

Properties with a type, ending to '*' are required.


## Evaluator: Schnell.Ai.Actions.Classification.ClassificationEvaluator

Use the classification-evaluator to classify data-rows with a given model.

### Configuration

| Name       | Type    | Description                            | Default |
|------------|---------|----------------------------------------|---------|
| Model      | string* | Name of model to train                 |         |
| InputData  | string* | Name of dataset to use for evaluation  |         |
| ResultData | string* | Name of dataset to use for result-data |         |

Properties with a type, ending to '*' are required.