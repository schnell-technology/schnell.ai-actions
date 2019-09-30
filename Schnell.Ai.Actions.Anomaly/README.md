# schnell.AI Actions (Anomaly-Detection)

This module provides all components required for anomaly-detection.


## Installation
``` powershell
schnell.ai module-i Schnell.Ai.Actions.Anomaly 0.9.1
```

## Trainer: Schnell.Ai.Actions.Anomaly.AnomalyTrainer

Use the anomaly-trainer to train a new model.

### Configuration

| Name         | Type    | Description                              | Default |
|--------------|---------|------------------------------------------|---------|
| Model        | string* | Name of model to train                   |         |
| TrainingData | string* | Name of dataset to use for training-data |         |
| UseCaching   | boolean | Use Caching of the data                  | false   |

Properties with a type, ending to '*' are required.

## Tester: Schnell.Ai.Actions.Anomaly.AnomalyTester

Use the anomaly-tester to test a model against test-data.

### Configuration

| Name           | Type    | Description                            | Default |
|----------------|---------|----------------------------------------|---------|
| Model          | string* | Name of model to train                 |         |
| TestData       | string* | Name of dataset to use for test-data   |         |
| TestResultData | string* | Name of dataset to use for result-data |         |

Properties with a type, ending to '*' are required.


## Evaluator: Schnell.Ai.Actions.Anomaly.AnomalyTester

Use the anomaly-evaluator to detect anomalies with a model and evaluation-data.

### Configuration

| Name       | Type    | Description                            | Default |
|------------|---------|----------------------------------------|---------|
| Model      | string* | Name of model to train                 |         |
| InputData  | string* | Name of dataset to use for evaluation  |         |
| ResultData | string* | Name of dataset to use for result-data |         |

Properties with a type, ending to '*' are required.