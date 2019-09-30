using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Schnell.Ai.Sdk;
using Schnell.Ai.Sdk.Configuration;

namespace Schnell.Ai.Actions.Clustering
{
    /// <summary>
    /// Configuration for cluster-trainer
    /// </summary>
    [DataContract]
    public class ClusterTrainerConfiguration : Schnell.Ai.Sdk.Components.TrainerConfiguration
    {
        /// <summary>
        /// Numbers of clusters to be rationed
        /// </summary>
        [DataMember(Name = "NumberOfClusters")]
        public Int64 NumberOfClusters { get; set; }
    }

    /// <summary>
    /// Trainer for cluster-models
    /// </summary>
    public class ClusterTrainer : Schnell.Ai.Sdk.Components.TrainerBase
    {
        MLContext mlContext;

        protected ConfigurationHandler<ClusterTrainerConfiguration> ConfigurationHandler { get; set; }
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

            var concateFields = trainingData.FieldDefinitions.Where(f => f.FieldType == Sdk.Definitions.FieldDefinition.FieldTypeEnum.Feature).Select(f => f.Name);
            var dataProcessPipeline = mlContext.Transforms.Concatenate("__Features", concateFields.ToArray());

            var trainer = mlContext.Clustering.Trainers.KMeans(featureColumnName: "__Features", numberOfClusters: (int)ConfigurationHandler.Configuration.NumberOfClusters);
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            this.Log.Write(Sdk.Logging.LogEntry.LogType.Info, $"Preparation completed, start training for {ConfigurationHandler.Configuration.NumberOfClusters} clusters.");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, model.Filepath);
        }

        protected override void OnBuilt()
        {
            base.OnBuilt();
            ConfigurationHandler = new ConfigurationHandler<ClusterTrainerConfiguration>(this.Definition);
            mlContext = new MLContext(seed: 1);
        }


    }

}
