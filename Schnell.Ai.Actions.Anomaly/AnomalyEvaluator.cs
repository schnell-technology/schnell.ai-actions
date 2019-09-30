using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Schnell.Ai.Sdk;
using Schnell.Ai.Sdk.Configuration;

namespace Schnell.Ai.Actions.Anomaly
{
    /// <summary>
    /// Evaluator for anomaly-models
    /// </summary>
    public class AnomalyEvaluator : Schnell.Ai.Sdk.Components.EvaluatorBase
    {
        MLContext mlContext;
        public override async Task Process(PipelineContext context)
        {
            var modelData = context.Artifacts.GetModel(ConfigurationHandler.Configuration.Model);
            var evaluationData = context.Artifacts.GetDataSet(ConfigurationHandler.Configuration.InputData);
            var resultData = context.Artifacts.GetDataSet(ConfigurationHandler.Configuration.ResultData);

            if (evaluationData == null)
            {
                this.Log.Write(Sdk.Logging.LogEntry.LogType.Fatal, "InputData not available");
                return;
            }

            if (resultData == null)
            {
                this.Log.Write(Sdk.Logging.LogEntry.LogType.Fatal, "ResultData not available");
                return;
            }

            if (modelData == null)
            {
                this.Log.Write(Sdk.Logging.LogEntry.LogType.Fatal, "Model not available");
                return;
            }


            var evDataContent = await evaluationData.GetContent();
            var results = GetResults(context, modelData, evaluationData, resultData, evDataContent).ToList();
            await resultData.SetContent(results);
        }

        private IEnumerable<IDictionary<string, object>> GetResults(
            PipelineContext context,
            Sdk.Definitions.ModelDefinition modelData,
            Sdk.DataSets.DataSet evaluationData,
            Sdk.DataSets.DataSet resultData,
            IEnumerable<IDictionary<string, object>> evDataContent)
        {
            ITransformer model = mlContext.Model.Load(modelData.Filepath, out var modelInputSchema);

            var evaluationDataType = Sdk.Utilities.FieldDefinitionToTypeCompiler.CreateTypeFromFielDefinitions(evaluationData.FieldDefinitions, "Context.Data.DynamicType_" + Guid.NewGuid().ToString("N"));
            var resultDataType = Sdk.Utilities.FieldDefinitionToTypeCompiler.CreateTypeFromFielDefinitions(evaluationData.FieldDefinitions, "Context.Data.DynamicType_" + Guid.NewGuid().ToString("N"));

            this.Log.Write(Sdk.Logging.LogEntry.LogType.Info, "Initialize and cache evaluation-data");
            var evaluationDataDict = evDataContent.ToList();
            var evaluationDataList = (evaluationDataDict).Select(s => Shared.Helper.ObjectDictionaryMapper.GetObject(evaluationDataType, s));
            var typedEvaluationDataList = Shared.Helper.Enumerable.CastEnumerable(evaluationDataList, evaluationDataType);
            var evaluationDataView = (IDataView)(Shared.Helper.Reflection.MakeGenericAndInvoke(mlContext.Data, "LoadFromEnumerable", evaluationDataType, typedEvaluationDataList, null));

            this.Log.Write(Sdk.Logging.LogEntry.LogType.Info, "Begin analyzing evaluation-data");
            var transformedData = model.Transform(evaluationDataView);
            List<AnomalyEvaluationData> predictions = mlContext.Data.CreateEnumerable<AnomalyEvaluationData>(transformedData, false).ToList();
            this.Log.Write(Sdk.Logging.LogEntry.LogType.Info, "Anomalies in evaluation-data predicted...");

            var processed = 0;
            for(int i = 0;i< evaluationDataDict.Count; i++)
            {
                var ev = evaluationDataDict[i];                                
                var score = predictions[i].Prediction.FirstOrDefault();
                var resultDict = DictionaryHelper.MergeDictionaryAndResults(ev, resultData.FieldDefinitions, score);
                processed++;
                this.Log.Progress(currentValue: processed);

                yield return resultDict;
                
            }
        }

        protected override void OnBuilt()
        {
            base.OnBuilt();
            mlContext = new MLContext(seed: 1);
        }
    }


    internal class AnomalyEvaluationData
    {        
        public float[] Prediction;
    }
}
