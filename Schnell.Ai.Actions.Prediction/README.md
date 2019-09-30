# schnell.AI Actions (Prediction)

This module provides all components required for data-prediction.


## Installation
``` powershell
schnell.ai module-i Schnell.Ai.Actions.Prediction 0.9.1
```

## Trainer: Schnell.Ai.Actions.Prediction.PredictionTrainer

Use the prediction-trainer to train a new model.

### Configuration

| Name                          | Type    | Description                              | Default |
|-------------------------------|---------|------------------------------------------|---------|
| Model                         | string* | Name of model to train                   |         |
| TrainingData                  | string* | Name of dataset to use for training-data |         |
| UseCaching                    | bool    | Enable caching for training-data         | false   |
| Sdca                          | object  | Configuration for Sdca-Algorithm         |         |
| Sdca.MaximumIterations        | int     | Maximum iterations for training          |         |
| Poisson                       | object  | Configuration for Poisson-Algorithm      |         |
| Poisson.OptimizationTolerance | float   | Tolerance                                |         |
| Poisson.HistorySize           | int     | Amount of data to use for history        |         |
| Poisson.NoNegativity          | bool    | Disable negative results                 |         |
| FastTree                      | object  | Configuration for FastTree-Algorithm     |         |
| FastTree.NumberOfLeaves       | int     | Amount of leaves                         |         |
| FastTree.NumberOfTrees        | int     | Amount of trees                          |         |

Properties with a type, ending to '*' are required.

## Tester: Schnell.Ai.Actions.Prediction.PredictionTester

Use the prediction-tester to test a model against test-data.

### Configuration

| Name           | Type    | Description                            | Default |
|----------------|---------|----------------------------------------|---------|
| Model          | string* | Name of model to train                 |         |
| TestData       | string* | Name of dataset to use for test-data   |         |
| TestResultData | string* | Name of dataset to use for result-data |         |

Properties with a type, ending to '*' are required.

## Evaluator: Schnell.Ai.Actions.Prediction.PredictionEvaluator

Use the clustering-evaluator to cluster data-rows with a given model.

### Configuration

| Name       | Type    | Description                            | Default |
|------------|---------|----------------------------------------|---------|
| Model      | string* | Name of model to train                 |         |
| InputData  | string* | Name of dataset to use for evaluation  |         |
| ResultData | string* | Name of dataset to use for result-data |         |

Properties with a type, ending to '*' are required.