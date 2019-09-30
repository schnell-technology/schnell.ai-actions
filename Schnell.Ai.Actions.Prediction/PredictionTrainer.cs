using Microsoft.ML;
using Microsoft.ML.Data;
using Schnell.Ai.Sdk;
using Schnell.Ai.Sdk.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace Schnell.Ai.Actions.Prediction
{
    /// <summary>
    /// Configuration for prediction-trainer
    /// </summary>
    [DataContract]
    public class PredictionTrainerConfiguration : Schnell.Ai.Sdk.Components.TrainerConfiguration
    {
        [DataMember(Name = "UseCaching")]
        public bool UseCaching { get; set; } = true;

        [DataMember(Name = "Sdca")]
        public SdcaSettings Sdca { get; set; }

        [DataMember(Name = "Poisson")]
        public PoissonSettings Poisson { get; set; }

        [DataMember(Name = "FastTree")]
        public FastTreeSettings FastTree { get; set; }
    }

    [DataContract]
    public class SdcaSettings
    {
        [DataMember]
        public int? MaximumIterations { get; set; }
    }


    [DataContract]
    public class PoissonSettings
    {
        [DataMember]
        public float? OptimizationTolerance { get; set; }

        [DataMember]
        public int? HistorySize { get; set; }

        [DataMember]
        public bool NoNegativity { get; set; }
    }

    [DataContract]
    public class FastTreeSettings
    {
        [DataMember]
        public int NumberOfLeaves { get; set; }

        [DataMember]
        public int NumberOfTrees { get; set; }
    }    

    /// <summary>
    /// Trainer for classification-models
    /// </summary>
    public class PredictionTrainer : Schnell.Ai.Sdk.Components.TrainerBase
    {
        MLContext mlContext;

        protected ConfigurationHandler<PredictionTrainerConfiguration> ConfigurationHandler { get; set; }
        public override IConfigurationHandler Configuration => ConfigurationHandler;

        public override async Task Process(PipelineContext context)
        {
            var trainingData = context.Artifacts.GetDataSet(ConfigurationHandler.Configuration.TrainingData);
            var model = context.Artifacts.GetModel(ConfigurationHandler.Configuration.Model);
            var trainingDataType = Sdk.Utilities.FieldDefinitionToTypeCompiler.CreateTypeFromFielDefinitions(trainingData.FieldDefinitions, "Context.Data.DynamicType_" + Guid.NewGuid().ToString("N"));

            if (trainingData == null)
            {
                this.Log.Write(Sdk.Logging.LogEntry.LogType.Fatal, "TrainingData not available");
                return;
            }

            if (model == null)
            {
                this.Log.Write(Sdk.Logging.LogEntry.LogType.Fatal, "Model not available");
                return;
            }

            if (trainingDataType == null)
            {
                this.Log.Write(Sdk.Logging.LogEntry.LogType.Fatal, "Could not build type for training-data");
                return;
            }

            var trainingDataList = (await trainingData.GetContent()).Select(s => Shared.Helper.ObjectDictionaryMapper.GetObject(trainingDataType, s));
            var typedTrainingDataList = Shared.Helper.Enumerable.CastEnumerable(trainingDataList, trainingDataType);
            var trainingDataView = (IDataView)(Shared.Helper.Reflection.MakeGenericAndInvoke(mlContext.Data, "LoadFromEnumerable", trainingDataType, typedTrainingDataList, null));

            var labelField = trainingData.FieldDefinitions.FirstOrDefault(f => f.FieldType == Sdk.Definitions.FieldDefinition.FieldTypeEnum.Label)?.Name ?? "Label";
            var concateFields = trainingData.FieldDefinitions.Where(f => f.FieldType == Sdk.Definitions.FieldDefinition.FieldTypeEnum.Feature).Select(f => f.Name);

            var dataProcessPipeline = mlContext.Transforms.Concatenate(Constants.procPipeFeatures, concateFields.ToArray()).Append(mlContext.Transforms.CopyColumns(Constants.procPipeLabel, labelField));

            if (ConfigurationHandler.Configuration.UseCaching)
                dataProcessPipeline = dataProcessPipeline.AppendCacheCheckpoint(mlContext);

            IEstimator<ITransformer> trainer = null;

            if (ConfigurationHandler.Configuration.Sdca != null)
            {
                trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: Constants.procPipeLabel, featureColumnName: Constants.procPipeFeatures, maximumNumberOfIterations: ConfigurationHandler.Configuration.Sdca.MaximumIterations);
            }
            else if (ConfigurationHandler.Configuration.Poisson != null)
            {
                trainer = mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: Constants.procPipeLabel, featureColumnName: Constants.procPipeFeatures);
            }
            else if (ConfigurationHandler.Configuration.FastTree != null)
            {
                trainer = mlContext.Regression.Trainers.FastTree(labelColumnName: Constants.procPipeLabel, featureColumnName: Constants.procPipeFeatures, numberOfLeaves: ConfigurationHandler.Configuration.FastTree.NumberOfLeaves, numberOfTrees: ConfigurationHandler.Configuration.FastTree.NumberOfTrees);
            }
            else
            {
                trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: Constants.procPipeLabel, featureColumnName: Constants.procPipeFeatures);
            }


            var trainingPipeline = dataProcessPipeline.Append(trainer);

            this.Log.Write(Sdk.Logging.LogEntry.LogType.Info, $"Preparation completed, start classification training.");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, model.Filepath);
        }

        protected override void OnBuilt()
        {
            base.OnBuilt();
            ConfigurationHandler = new ConfigurationHandler<PredictionTrainerConfiguration>(this.Definition);
            mlContext = new MLContext(seed: 1);
        }


    }
}
