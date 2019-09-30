using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Schnell.Ai.Sdk;
using Schnell.Ai.Sdk.Components;

namespace Schnell.Ai.Actions.Classification
{
    /// <summary>
    /// Tester for classification-models
    /// </summary>
    public class ClassificationTester : Schnell.Ai.Sdk.Components.TesterBase
    {
        MLContext mlContext;
        protected override async Task<TestResult> ProcessTest(PipelineContext context)
        {
            var modelData = context.Artifacts.GetModel(ConfigurationHandler.Configuration.Model);
            var testData = context.Artifacts.GetDataSet(ConfigurationHandler.Configuration.TestData);

            if (modelData == null)
            {
                this.Log.Write(Sdk.Logging.LogEntry.LogType.Fatal, "Model not available");
                return null;
            }

            if (testData == null)
            {
                this.Log.Write(Sdk.Logging.LogEntry.LogType.Fatal, "InputData not available");
                return null;
            }

            var testDataType = Sdk.Utilities.FieldDefinitionToTypeCompiler.CreateTypeFromFielDefinitions(testData
                .FieldDefinitions, "Context.Data.DynamicType_" + Guid.NewGuid().ToString("N"));
            var testDataList = (await testData.GetContent()).Select(s => Shared.Helper.ObjectDictionaryMapper.GetObject(testDataType, s));
            var typedTestDataList = Shared.Helper.Enumerable.CastEnumerable(testDataList, testDataType);
            var testDataView = (IDataView)(Shared.Helper.Reflection.MakeGenericAndInvoke(mlContext.Data, "LoadFromEnumerable", testDataType, typedTestDataList, null));
            ITransformer model = mlContext.Model.Load(modelData.Filepath, out var modelInputSchema);

            var predictions = model.Transform(testDataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: Constants.procPipeLabel);

            var result = new TestResult();
            result.Score = (((1 - (float)metrics.LogLoss) + (float)metrics.MacroAccuracy) / 2);

            result.DetailResults.Add("LogLoss", metrics.LogLoss);
            result.DetailResults.Add("LogLossReduction", metrics.LogLossReduction);
            result.DetailResults.Add("MacroAccurancy", metrics.MacroAccuracy);
            result.DetailResults.Add("MicroAccurancy", metrics.MicroAccuracy);
            result.DetailResults.Add("TopKAccurancy", metrics.TopKAccuracy);
            result.DetailResults.Add("TopKPredictionCount", metrics.TopKPredictionCount);
            for (int i = 0;i<metrics.PerClassLogLoss.Count;i++)
            {
                result.DetailResults.Add($"LogLoss_Class{i}", metrics.PerClassLogLoss[i]);
            }

            return result;
        }

        protected override void OnBuilt()
        {
            base.OnBuilt();
            mlContext = new MLContext(seed: 1);
        }
    }
}
