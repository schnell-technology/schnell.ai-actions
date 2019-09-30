using Microsoft.ML;
using Schnell.Ai.Sdk;
using Schnell.Ai.Sdk.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace Schnell.Ai.Actions.Classification
{
    /// <summary>
    /// Configuration for classification-trainer
    /// </summary>
    [DataContract]
    public class ClassificationTrainerConfiguration : Schnell.Ai.Sdk.Components.TrainerConfiguration
    {
        [DataMember(Name = "UseCaching")]
        public bool UseCaching { get; set; } = true;

        [DataMember(Name = "Algorithm")]
        public ClassificationAlgorithm Algorithm { get; set; }
    }

    /// <summary>
    /// Multiclass-classification-algorithm type
    /// </summary>
    [DataContract]
    public enum ClassificationAlgorithm
    {
        SDCA_MaximumEntropy = 0,
        LBFGS_MaximumEntropy = 1
    }

    /// <summary>
    /// Trainer for classification-models
    /// </summary>
    public class ClassificationTrainer : Schnell.Ai.Sdk.Components.TrainerBase
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
            var concateFields = trainingData.FieldDefinitions.Where(f => f.FieldType == Sdk.Definitions.FieldDefinition.FieldTypeEnum.Feature).Select(f => f.Name);

            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: Constants.procPipeLabel, inputColumnName: labelField)
                                      .Append(mlContext.Transforms.Concatenate(Constants.procPipeFeatures, concateFields.ToArray()));

            if(ConfigurationHandler.Configuration.UseCaching)
                dataProcessPipeline = dataProcessPipeline.AppendCacheCheckpoint(mlContext);

            Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> trainer = null;

            if (ConfigurationHandler.Configuration.Algorithm == ClassificationAlgorithm.SDCA_MaximumEntropy)
            {
                trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(featureColumnName: Constants.procPipeFeatures, labelColumnName: Constants.procPipeLabel)
                                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: Constants.procPipePredicted, inputColumnName: Constants.procPipeLabel));
            }
            else if(ConfigurationHandler.Configuration.Algorithm == ClassificationAlgorithm.LBFGS_MaximumEntropy)
            {
                trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(featureColumnName: Constants.procPipeFeatures, labelColumnName: Constants.procPipeLabel)
                                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: Constants.procPipePredicted, inputColumnName: Constants.procPipeLabel));
            }


            var trainingPipeline = dataProcessPipeline.Append(trainer);

            this.Log.Write(Sdk.Logging.LogEntry.LogType.Info, $"Preparation completed, start classification training.");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

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
