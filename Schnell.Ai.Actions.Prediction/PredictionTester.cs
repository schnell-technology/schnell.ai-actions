using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Schnell.Ai.Sdk;
using Schnell.Ai.Sdk.Components;

namespace Schnell.Ai.Actions.Prediction
{
    /// <summary>
    /// Tester for classification-models
    /// </summary>
    public class PredictionTester : Schnell.Ai.Sdk.Components.TesterBase
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
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: Constants.procPipeLabel);

            var result = new TestResult();
            result.Score = (1 - (float)metrics.RSquared);

            result.DetailResults.Add("LossFunction", metrics.LossFunction);
            result.DetailResults.Add("MeanAbsoluteError", metrics.MeanAbsoluteError);
            result.DetailResults.Add("MeanSquaredError", metrics.MeanSquaredError);
            result.DetailResults.Add("RootMeanSquaredError", metrics.RootMeanSquaredError);
            result.DetailResults.Add("RSquared", metrics.RSquared);

            return result;
        }

        protected override void OnBuilt()
        {
            base.OnBuilt();
            mlContext = new MLContext(seed: 1);
        }
    }
}
