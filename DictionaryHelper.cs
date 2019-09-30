using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Helper for dictionary operations
/// </summary>
public static class DictionaryHelper
{
    public static IDictionary<string, object> MergeDictionaryAndResults(
        IDictionary<string, object> data,
        IEnumerable<Schnell.Ai.Sdk.Definitions.FieldDefinition> fieldDefinitions,
        float? score = null,
        string predictedLabel = null
        )
    {
        var result = new Dictionary<string, object>();
        fieldDefinitions.ToList().ForEach(
            f =>
            {
                if (f.FieldType == Schnell.Ai.Sdk.Definitions.FieldDefinition.FieldTypeEnum.Label)
                {
                    result[f.Name] = predictedLabel;
                }
                else if (f.FieldType == Schnell.Ai.Sdk.Definitions.FieldDefinition.FieldTypeEnum.Score)
                {
                    result[f.Name] = score ?? 0;
                }
                else
                {
                    if (data.ContainsKey(f.Name))
                    {
                        result[f.Name] = data[f.Name];
                    }
                    else
                    {
                        result[f.Name] = null;
                    }
                }
            }
        );
        return result;
    }
}
