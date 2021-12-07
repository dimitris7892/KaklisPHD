using System;

namespace ConsModelComparison_webapi
{
    public class CompareConsModels
    {
        public double Speed{ get; set; }
        public double Size{ get; set; }
        public double NeuralDT_Acc{ get; set; }

        public double StatsDT_Acc{ get; set; }
        
        public double Actual_Avg{ get; set; }
        
        public double NeuralDT_Avg{ get; set; }
        public double Stats_Avg{ get; set; }
        public double NeuralDT_RMSE{ get; set; }
        
        public double Stats_RMSE { get; set; }
       
        
    }
}
