# schnell.AI Actions (Clustering)

This module provides all components required for data-clustering.

## Trainer: Schnell.Ai.Actions.Clustering.ClusterTrainer

Use the clustering-trainer to train a new model.

### Configuration

| Name             | Type    | Description                              | Default |
|------------------|---------|------------------------------------------|---------|
| Model            | string* | Name of model to train                   |         |
| TrainingData     | string* | Name of dataset to use for training-data |         |
| NumberOfClusters | int*    | Numbers of clusters you want to use      | 0       |

Properties with a type, ending to '*' are required.


## Evaluator: Schnell.Ai.Actions.Clustering.ClusterEvaluator

Use the clustering-evaluator to cluster data-rows with a given model.

### Configuration

| Name       | Type   | Description                            | Default |
|------------|--------|----------------------------------------|---------|
| Model      | string* | Name of model to train                 |         |
| InputData  | string* | Name of dataset to use for evaluation  |         |
| ResultData | string* | Name of dataset to use for result-data |         |

Properties with a type, ending to '*' are required.