﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Schnell.Ai.Sdk;
using Schnell.Ai.Sdk.Configuration;

namespace Schnell.Ai.Actions.Prediction
{
    /// <summary>
    /// Evaluator for prediction-models
    /// </summary>
    public class PredictionEvaluator : Schnell.Ai.Sdk.Components.EvaluatorBase
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

            var loadMethodGeneric = mlContext.Model.GetType().GetMethods().First(method => method.Name == nameof(mlContext.Model.CreatePredictionEngine) && method.IsGenericMethod);
            var loadMethod = loadMethodGeneric.MakeGenericMethod(evaluationDataType, typeof(PredictionEvaluationData));
            var predEngine = loadMethod.Invoke(mlContext.Model, new object[] { model, null, null, null });
            var predicter = predEngine.GetType().GetMethods().First(p => p.Name == "Predict" && p.GetParameters().Count() == 1);

            var processed = 0;
            foreach (var ev in evDataContent)
            {
                var typedEv = Shared.Helper.ObjectDictionaryMapper.GetObject(evaluationDataType, ev);
                var result = predicter.Invoke(predEngine, new[] { typedEv }) as PredictionEvaluationData;
                if (result != null)
                {                    
                    var resultDict = DictionaryHelper.MergeDictionaryAndResults(ev, resultData.FieldDefinitions, result.Score);
                    processed++;
                    this.Log.Progress(currentValue: processed);

                    yield return resultDict;
                }
            }
        }

        protected override void OnBuilt()
        {
            base.OnBuilt();
            mlContext = new MLContext(seed: 1);
        }
    }


    internal class PredictionEvaluationData
    {
        [ColumnName("Score")]
        public float Score;
    }
}
