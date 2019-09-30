using System;
using System.Collections.Generic;
using System.Text;

namespace Schnell.Ai.Actions.Anomaly
{
    class Constants
    {
        internal const string procPipeLabel = "SAI_Lbl";
        internal const string procPipeFeatures = "SAI_Ftr";
        internal const string procPipePredicted = "SAI_PLbl";

        internal const int PValueSize = 30;
        internal const int SeasonalitySize = 30;
        internal const int TrainingSize = 90;
        internal const int ConfidenceInterval = 98;
    }
}
