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

namespace Schnell.Ai.Actions.Anomaly
{
    /// <summary>
    /// Configuration for classification-trainer
    /// </summary>
    [DataContract]
    public class ClassificationTrainerConfiguration : Schnell.Ai.Sdk.Components.TrainerConfiguration
    {
        [DataMember]
        public bool UseCaching { get; set; } = false;
        [DataMember]
        public int ConfidenceInterval { get; set; } = 98;
        [DataMember]
        public int PValueSize { get; set; } = 30;
        [DataMember]
        public int TrainingWindowSize { get; set; } = 30;
        [DataMember]
        public int SeasonalityWindowSize { get; set; } = 30;

    }

    /// <summary>
    /// Trainer for anomaly-models
    /// </summary>
    public class AnomalyTrainer : Schnell.Ai.Sdk.Components.TrainerBase
    {
        MLContext mlContext;

        protected ConfigurationHandler<ClassificationTrainerConfiguration> ConfigurationHandler { get; set; }
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
            var concateFields = trainingData.FieldDefinitions.Where(f => f.FieldType == Sdk.Definitions.FieldDefinition.FieldTypeEnum.Feature).Select(f => f.Name).ToArray();

            var dataProcessPipeline = 
                mlContext.Transforms.Concatenate(Constants.procPipeFeatures, concateFields).Append(
                mlContext.Transforms.DetectSpikeBySsa(
                Constants.procPipePredicted,
                Constants.procPipeFeatures,
                confidence: this.ConfigurationHandler.Configuration.ConfidenceInterval,
                pvalueHistoryLength: this.ConfigurationHandler.Configuration.PValueSize,
                trainingWindowSize: this.ConfigurationHandler.Configuration.TrainingWindowSize,
                seasonalityWindowSize: this.ConfigurationHandler.Configuration.SeasonalityWindowSize));

            if (ConfigurationHandler.Configuration.UseCaching)
                dataProcessPipeline = dataProcessPipeline.AppendCacheCheckpoint(mlContext);



            this.Log.Write(Sdk.Logging.LogEntry.LogType.Info, $"Preparation completed, start anomaly training.");
            var trainedModel = dataProcessPipeline.Fit(trainingDataView);

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, model.Filepath);
        }

        protected override void OnBuilt()
        {
            base.OnBuilt();
            ConfigurationHandler = new ConfigurationHandler<ClassificationTrainerConfiguration>(this.Definition);
            mlContext = new MLContext(seed: 1);
        }


    }
}
