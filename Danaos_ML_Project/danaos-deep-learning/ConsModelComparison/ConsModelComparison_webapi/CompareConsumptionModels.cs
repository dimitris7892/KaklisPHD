using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Configuration;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace ConsModelComparison
{
    public class CompareConsumptionModels
    {
        private static IConfiguration _config;
        
        public CompareConsumptionModels(IConfiguration config)
        {
            
            _config = config;
        }
        public void CompareModels(string connStr, string pathUnseenData, string pathConsProfileNeural, string pathWrite, bool interpolation)
        {   
            
            // Get unseen data
            
            string unseenDataStr = File.ReadAllText(pathUnseenData);
            UnseenDataSet cpn = JsonConvert.DeserializeObject<UnseenDataSet>(unseenDataStr);


            // ------- Retrieve model from DB -------
            //string vesselCode = cp_DT_Neural.CP.ConsumptionProfile.vessel_code;
            //// Get ConsProfile from DB
            //ConsumptionProfileFull_3 cp_Stats = RetrieveConsProfileFromDB(vesselCode, connStr.ProdRoutingDB);
            //if (cp_Stats == null)
            //    return;
            // ------- Retrieve model from file ------
            string consProfileStr_Stats = File.ReadAllText(pathConsProfileNeural);
            ConsumptionProfileFull_3 cp_Stats = JsonConvert.DeserializeObject<ConsumptionProfileFull_3>(consProfileStr_Stats);
            // ---------------------------------------
            ConsProf_DT cp_DT_Neural = new ConsProf_DT(cp_Stats);

            // Compare models to real consumption
            //bool interpolation = true;

            Comparison comp = new Comparison("Neural", "Stats");
            comp.Compare(cpn.ConsumptionProfile_Dataset.data, cp_DT_Neural, interpolation,pathWrite);
            //comp.CompareSpeed(cpn.ConsumptionProfile_Dataset.data, cp_DT_Neural, cp_DT_Stats, interpolation);
            //List<ComparisonSpeed> compareSpeedList = comp.CreateCompareSpeedList();
            //if (interpolation)
            //{
                
            //    string compareSpeedStr = "Size,NeuralDT Acc, StatsDT Acc,Actual Avg,NeuralDT Avg,Stats Avg,NeuralDT RMSE,Stats RMSE \n";
            //    compareSpeedStr = compareSpeedStr + comp.sizeOfDataset + "," + comp.percDiffModel1 + "," + comp.percDiffModel2 + "," + 1440 * comp.avgActual + "," + 1440 * comp.avgModel1 + "," + 1440 * comp.avgModel2 + "," + 1440 * comp.rmseModel1 + "," + 1440 * comp.rmseModel2 + "\n\n\n";
            //    compareSpeedStr = compareSpeedStr + "Speed,Size,NeuralDT Acc,StatsDT Acc,Actual Avg,NeuralDT Avg,Stats Avg,NeuralDT RMSE,Stats RMSE\n";
            //    for (int i = 0; i < compareSpeedList.Count; i++)
            //        compareSpeedStr = compareSpeedStr + compareSpeedList[i].speed + "," + compareSpeedList[i].sizeOfDataset + "," + compareSpeedList[i].percDiffModel1 + "," + compareSpeedList[i].percDiffModel2 + "," + 1440 * compareSpeedList[i].avgActual + "," + 1440 * compareSpeedList[i].avgModel1 + "," + 1440 * compareSpeedList[i].avgModel2 + "," + 1440 * compareSpeedList[i].rmseModel1 + "," + 1440 * compareSpeedList[i].rmseModel2 + "\n";

            //    string pathWriteInterp = pathWrite + "_interp_" + DateTime.Now.ToString("yyyyMMdd_HHmm") + ".csv";
            //    File.WriteAllText(pathWriteInterp, compareSpeedStr);
            //}

            ////interpolation = false;
            /////////////////////////////////////////////////////////////////////////Without Interpolation
            //Comparison compAvg = new Comparison("Neural", "Stats");
            //compAvg.Compare(cpn.ConsumptionProfile_Dataset.data, cp_DT_Neural, cp_DT_Stats, interpolation, pathWrite);
            //compAvg.CompareSpeed(cpn.ConsumptionProfile_Dataset.data, cp_DT_Neural, cp_DT_Stats, interpolation);
            //List<ComparisonSpeed> compareSpeedListAvg = compAvg.CreateCompareSpeedList();
            //if (!interpolation)
            //{    
            //    string compareSpeedStr = "Size,NeuralDT Acc, StatsDT Acc,Actual Avg,NeuralDT Avg,Stats Avg,NeuralDT RMSE,Stats RMSE \n";
            //    compareSpeedStr = compareSpeedStr + compAvg.sizeOfDataset + "," + compAvg.percDiffModel1 + "," + compAvg.percDiffModel2 + "," + 1440 * compAvg.avgActual + "," + 1440 * compAvg.avgModel1 + "," + 1440 * compAvg.avgModel2 + "," + 1440 * compAvg.rmseModel1 + "," + 1440 * compAvg.rmseModel2 + "\n\n\n";
            //    compareSpeedStr = compareSpeedStr + "Speed,Size,NeuralDT Acc,StatsDT Acc,Actual Avg,NeuralDT Avg,Stats Avg,NeuralDT RMSE,Stats RMSE\n";
            //    for (int i = 0; i < compareSpeedListAvg.Count; i++)
            //        compareSpeedStr = compareSpeedStr + compareSpeedListAvg[i].speed + "," + compareSpeedListAvg[i].sizeOfDataset + "," + compareSpeedListAvg[i].percDiffModel1 + "," + compareSpeedListAvg[i].percDiffModel2 + "," + 1440 * compareSpeedListAvg[i].avgActual + "," + 1440 * compareSpeedListAvg[i].avgModel1 + "," + 1440 * compareSpeedListAvg[i].avgModel2 + "," + 1440 * compareSpeedListAvg[i].rmseModel1 + "," + 1440 * compareSpeedListAvg[i].rmseModel2 + "\n";

            //    string pathWriteSimple = pathWrite + "_simple_" + DateTime.Now.ToString("yyyyMMdd_HHmm") + ".csv";
            //    File.WriteAllText(pathWriteSimple, compareSpeedStr);
            //}

            //return  interpolation ? compareSpeedList : compareSpeedListAvg;
        }

        /*public ConsumptionProfileFull_3 RetrieveConsProfileFromDB(string tempVesselCode, string sqlConnStr)
        {
            ConsumptionProfileFull_3 consProfFull = null;
            using (SqlConnection sqlConn = new SqlConnection(sqlConnStr))
            {
                sqlConn.Open();
                using (SqlDataAdapter adapter1 = new SqlDataAdapter())
                {
                    SqlCommand selectCommand1 = new SqlCommand("SELECT profile FROM ConsProfileNew WHERE vesselCode = '" + tempVesselCode + "'", sqlConn);
                    adapter1.SelectCommand = selectCommand1;
                    using (DataSet data = new DataSet())
                    {
                        adapter1.Fill(data, "Data");
                        int numberOfData = data.Tables["Data"].Rows.Count;
                        if (numberOfData > 0)
                        {
                            string profileStr = data.Tables["Data"].Rows[0]["profile"].ToString();
                            consProfFull = JsonConvert.DeserializeObject<ConsumptionProfileFull_3>(profileStr);
                        }
                    }
                }
                sqlConn.Close();
            }

            return consProfFull;
        }*/

        public bool CheckValidityOfModel(ConsumptionProfileFull_3 cp, int cellsPerTable)
        {
            bool valid = true;

            for (int i = 0; i < cp.ConsumptionProfile.consProfile.Count; i++)
                if (cp.ConsumptionProfile.consProfile[i].cells.Count != cellsPerTable)
                {
                    //Console.WriteLine("Validity check " + i.ToString() + ": The table does not have " + cellsPerTable.ToString() + " cells.");
                    valid = true;
                }

            if (cp.ConsumptionProfile.vessel_code.Length != 7)
            {
                Console.WriteLine("Validity check - Vessel Code: It is not 7 characters (IMO number)");
                valid = false;
            }

            if (!IsDigitsOnly(cp.ConsumptionProfile.vessel_code))
            {
                Console.WriteLine("Validity check - Vessel Code: It is not an IMO number (it does not contain 7 digits)");
                valid = false;
            }

            return valid;
        }

        public bool IsDigitsOnly(string str)
        {
            foreach (char c in str)
            {
                if (c < '0' || c > '9')
                    return false;
            }

            return true;
        }
    }

    public class Comparison
    {
        public string model1 { set; get; }
        public string model2 { set; get; }
        public double sizeOfDataset { set; get; }
        public double consumptionActual { set; get; }
        public double consumptionModel1 { set; get; }
        public double consumptionModel2 { set; get; }
        public double avgActual { set; get; }
        public double avgModel1 { set; get; }
        public double avgModel2 { set; get; }
        public double mseModel1 { set; get; }
        public double mseModel2 { set; get; }
        public double rmseModel1 { set; get; }
        public double rmseModel2 { set; get; }
        public double diffModel1 { set; get; }
        public double diffModel2 { set; get; }
        public double percDiffModel1 { set; get; }
        public double percDiffModel2 { set; get; }
        public Dictionary<double, ComparisonSpeed> compareSpeedDict { set; get; }

        public double averageActualPerDay { set; get; }
        public double averageModel1PerDay { set; get; }
        public double averageModel2PerDay { set; get; }
        public double mseModel1PerDay { set; get; }
        public double mseModel2PerDay { set; get; }
        public double rmseModel1PerDay { set; get; }
        public double rmseModel2PerDay { set; get; }

        //double convertFromDayToMin = 1440;
        double convertFromDayToHour = 24;

        public Comparison(string model1, string model2)
        {
            this.model1 = model1;
            this.model2 = model2;
            compareSpeedDict = new Dictionary<double, ComparisonSpeed>();
        }

        public void Compare(List<UnseenDataPoint> data, ConsProf_DT cp_DT_Neural, bool interpolation,string pathWrite)
        {   
            List<double> speeds = new List<double>();
            List<double> drafts = new List<double>();
            List<double> windSpeeds = new List<double>();
            List<double> actualCons = new List<double>();
            List<double> statsCons = new List<double>();

            sizeOfDataset = data.Count;
            for (int i = 0; i < data.Count; i++)
            {
                // Per minute
                double tempActual = (data[i].cons / convertFromDayToHour);
                double tempModel1 = 0;
                if (interpolation)
                {
                    tempModel1 = cp_DT_Neural.CalculateConsumptionPerMinInterpolation(data[i]);
                }
                else
                {
                    tempModel1 = cp_DT_Neural.CalculateConsumptionPerMinSimple(data[i]);
                }
                consumptionActual += tempActual; // This is MT/min for each min
                consumptionModel1 += tempModel1; // This is MT/min for each min

                // Per day
                double tempActualPerDay = tempActual * convertFromDayToHour;
                double tempModel1PerDay = tempModel1 * convertFromDayToHour;

                speeds.Add(data[i].stw);
                drafts.Add(data[i].draft);
                actualCons.Add(data[i].cons);
                windSpeeds.Add(data[i].windMperS);
                statsCons.Add(tempModel1PerDay);
            }
            string compareSpeedStr = "FOC_pred,FOC_act,stw,draft,windMperS\n";
            for (int i = 0; i < data.Count; i++)
                    compareSpeedStr = compareSpeedStr + statsCons[i] + "," + actualCons[i] + "," + speeds[i] +
                    "," + drafts[i] + "," + windSpeeds[i]+"\n";

            string pathWriteInterp = pathWrite + "detailed_interp_" + DateTime.Now.ToString("yyyyMMdd_HHmm") + ".csv";
            File.WriteAllText(pathWriteInterp, compareSpeedStr);
            //// Per minute
            //avgActual = consumptionActual / sizeOfDataset;
            //avgModel1 = consumptionModel1 / sizeOfDataset;
            //avgModel2 = consumptionModel2 / sizeOfDataset;
            //mseModel1 = mseModel1 / sizeOfDataset;
            //mseModel2 = mseModel2 / sizeOfDataset;
            //rmseModel1 = Math.Sqrt(mseModel1);
            //rmseModel2 = Math.Sqrt(mseModel2);
            //diffModel1 = consumptionModel1 - consumptionActual;
            //diffModel2 = consumptionModel2 - consumptionActual;
            //percDiffModel1 = diffModel1 / consumptionActual;
            //percDiffModel2 = diffModel2 / consumptionActual;

            //// Per day
            //averageActualPerDay = averageActualPerDay / sizeOfDataset;
            //averageModel1PerDay = averageModel1PerDay / sizeOfDataset;
            //averageModel2PerDay = averageModel2PerDay / sizeOfDataset;
            //mseModel1PerDay = mseModel1PerDay / sizeOfDataset;
            //mseModel2PerDay = mseModel2PerDay / sizeOfDataset;
            //rmseModel1PerDay = Math.Sqrt(mseModel1PerDay);
            //rmseModel2PerDay = Math.Sqrt(mseModel2PerDay);
        }

        public void CompareSpeed(List<UnseenDataPoint> data, ConsProf_DT cp_DT_Neural, ConsProf_DT cp_DT_Stats, bool interpolation)
        {
            for (int i = 0; i < data.Count; i++)
            {
                double speed = cp_DT_Neural.IdentifySpeed(data[i]);
                if (!compareSpeedDict.ContainsKey(speed))
                {
                    ComparisonSpeed csTemp = new ComparisonSpeed(speed);
                    compareSpeedDict.Add(speed, csTemp);
                }
                ComparisonSpeed cs = compareSpeedDict[speed];
                cs.sizeOfDataset++;
                double tempActual = (data[i].cons / convertFromDayToHour);
                double tempModel1 = 0;
                double tempModel2 = 0;
                if (interpolation)
                {
                    tempModel1 = cp_DT_Neural.CalculateConsumptionPerMinInterpolation(data[i]);
                    tempModel2 = cp_DT_Stats.CalculateConsumptionPerMinInterpolation(data[i]);
                }
                else
                {
                    tempModel1 = cp_DT_Neural.CalculateConsumptionPerMinSimple(data[i]);
                    tempModel2 = cp_DT_Stats.CalculateConsumptionPerMinSimple(data[i]);
                }
                cs.consumptionActual += tempActual; // This is MT/min for each min
                cs.consumptionModel1 += tempModel1; // This is MT/min for each min
                cs.consumptionModel2 += tempModel2; // This is MT/min for each min
                cs.mseModel1 += Math.Pow(tempModel1 - tempActual, 2);
                cs.mseModel2 += Math.Pow(tempModel2 - tempActual, 2);
            }

            foreach (ComparisonSpeed cs in compareSpeedDict.Values)
            {
                cs.avgActual = cs.consumptionActual / cs.sizeOfDataset;
                cs.avgModel1 = cs.consumptionModel1 / cs.sizeOfDataset;
                cs.avgModel2 = cs.consumptionModel2 / cs.sizeOfDataset;
                cs.mseModel1 = cs.mseModel1 / cs.sizeOfDataset;
                cs.mseModel2 = cs.mseModel2 / cs.sizeOfDataset;
                cs.rmseModel1 = Math.Sqrt(cs.mseModel1);
                cs.rmseModel2 = Math.Sqrt(cs.mseModel2);
                cs.diffModel1 = cs.consumptionModel1 - cs.consumptionActual;
                cs.diffModel2 = cs.consumptionModel2 - cs.consumptionActual;
                cs.percDiffModel1 = cs.diffModel1 / cs.consumptionActual;
                cs.percDiffModel2 = cs.diffModel2 / cs.consumptionActual;
            }
        }

        public List<ComparisonSpeed> CreateCompareSpeedList()
        {
            List<ComparisonSpeed> compareSpeedList = new List<ComparisonSpeed>();

            foreach (var cs in compareSpeedDict.Values)
                compareSpeedList.Add(cs);

            compareSpeedList = compareSpeedList.OrderBy(o => o.speed).ToList();

            return compareSpeedList;
        }
    }

    public class ComparisonSpeed
    {
        public double speed { set; get; }
        public double sizeOfDataset { set; get; }
        public double consumptionActual { set; get; }
        public double consumptionModel1 { set; get; }
        public double consumptionModel2 { set; get; }
        public double avgActual { set; get; }
        public double avgModel1 { set; get; }
        public double avgModel2 { set; get; }
        public double mseModel1 { set; get; }
        public double mseModel2 { set; get; }
        public double rmseModel1 { set; get; }
        public double rmseModel2 { set; get; }
        public double diffModel1 { set; get; }
        public double diffModel2 { set; get; }
        public double percDiffModel1 { set; get; }
        public double percDiffModel2 { set; get; }

        public ComparisonSpeed(double speed)
        {
            this.speed = speed;
        }
    }

    public class ConsProf_DT
    {
        public Dictionary<string, double> DecisionTree_Type_3;
        public ConsumptionProfileFull_3 CP { set; get; }
        public List<BasicProfileSettingsPerDraft> listOfDraftsSpeeds { set; get; }
        public int TypeOfConsumptionModel = 4;
        NumberFormatInfo nfi = new NumberFormatInfo();

        public ConsProf_DT(ConsumptionProfileFull_3 _cp)
        {
            DecisionTree_Type_3 = new Dictionary<string, double>();
            listOfDraftsSpeeds = new List<BasicProfileSettingsPerDraft>();
            CP = _cp;
            ConsumptionProfile_3 consProf = CP.ConsumptionProfile;

            for (int i = 0; i < consProf.consProfile.Count; i++)
            {
                bool foundDraft = false;
                for (int j = 0; j < listOfDraftsSpeeds.Count; j++)
                    if (listOfDraftsSpeeds[j].draft == consProf.consProfile[i].draft)
                    {
                        foundDraft = true;
                        break;
                    }
                if (!foundDraft)
                {
                    BasicProfileSettingsPerDraft newDraft = new BasicProfileSettingsPerDraft(consProf.consProfile[i].draft);
                    listOfDraftsSpeeds.Add(newDraft);
                }
            }
            listOfDraftsSpeeds = listOfDraftsSpeeds.OrderBy(o => o.draft).ToList();

            for (int j = 0; j < listOfDraftsSpeeds.Count; j++)
            {
                listOfDraftsSpeeds[j].cp = new List<ConsProfileTable_3>();
                for (int i = 0; i < consProf.consProfile.Count; i++)
                    if (consProf.consProfile[i].draft == listOfDraftsSpeeds[j].draft)
                        listOfDraftsSpeeds[j].cp.Add(consProf.consProfile[i]);
                // Fill the list of speeds, and min/max speed values
                List<int> toRemove = new List<int>();
                listOfDraftsSpeeds[j].cp = listOfDraftsSpeeds[j].cp.OrderBy(o => o.speed).ToList();
                for (int i = 0; i < listOfDraftsSpeeds[j].cp.Count; i++)
                {
                    //== cellsPerTableValidation
                    if ((listOfDraftsSpeeds[j].cp[i].cells != null)  && ((i == 0) || (listOfDraftsSpeeds[j].listOfSpeeds[listOfDraftsSpeeds[j].listOfSpeeds.Count - 1] != listOfDraftsSpeeds[j].cp[i].speed)))
                        listOfDraftsSpeeds[j].listOfSpeeds.Add(listOfDraftsSpeeds[j].cp[i].speed);
                    else
                        toRemove.Add(i);
                    if (i == 0)
                        listOfDraftsSpeeds[j].minSpeed = listOfDraftsSpeeds[j].cp[i].speed;
                    else if (i == listOfDraftsSpeeds[j].cp.Count - 1)
                        listOfDraftsSpeeds[j].maxSpeed = listOfDraftsSpeeds[j].cp[i].speed;
                }

                for (int i = toRemove.Count - 1; i >= 0; i--) // For the case where a speed profile is identified twice (Exception)
                    listOfDraftsSpeeds[j].cp.RemoveAt(toRemove[i]);

                // Add consumption into decision tree dictionary and calculate max/min consumption
                for (int i = 0; i < listOfDraftsSpeeds[j].cp.Count; i++)
                    for (int k = 0; k < listOfDraftsSpeeds[j].cp[i].cells.Count; k++)
                    {
                        string key = listOfDraftsSpeeds[j].draft.ToString(nfi) + "_" + listOfDraftsSpeeds[j].cp[i].speed.ToString(nfi) + "_" + listOfDraftsSpeeds[j].cp[i].cells[k].windBFT + "_" + (listOfDraftsSpeeds[j].cp[i].cells[k].windDir - 1) + "_" + listOfDraftsSpeeds[j].cp[i].cells[k].swell;
                        DecisionTree_Type_3.Add(key, listOfDraftsSpeeds[j].cp[i].cells[k].cons);
                    }
            }
        }

        public double CalculateConsumptionPerMinSimple(UnseenDataPoint udp)
        {
            double consumptionPerMin = 0;
            double convertFromDayToHour = 24;
            // Choose draft
            int draftIndex = -1;
            double draftSel = -1000;
            double minDraftDiff = 1000;
            for (int i = 0; i < listOfDraftsSpeeds.Count; i++)
            {
                double diff = Math.Abs(udp.draft - listOfDraftsSpeeds[i].draft);
                if (diff < minDraftDiff)
                {
                    minDraftDiff = diff;
                    draftSel = listOfDraftsSpeeds[i].draft;
                    draftIndex = i;
                }
            }

            //Choose speed
            double speedTemp = udp.stw;
            int curspeedIndex = -1;
            // 0 : Exact speed
            // -1: Interpolate with previous index
            // +1: Interpolate with next index
            if (udp.stw < listOfDraftsSpeeds[draftIndex].minSpeed)
                speedTemp = listOfDraftsSpeeds[draftIndex].minSpeed;
            else if (speedTemp > listOfDraftsSpeeds[draftIndex].maxSpeed)
                speedTemp = listOfDraftsSpeeds[draftIndex].maxSpeed;
            // Find where it is in the list of speeds
            double adjustSpeedGroup = 0.2;
            for (int i = 0; i < listOfDraftsSpeeds[draftIndex].listOfSpeeds.Count; i++)
            {
                curspeedIndex = i;
                if (speedTemp <= listOfDraftsSpeeds[draftIndex].listOfSpeeds[i] + adjustSpeedGroup)
                    break;
            }
            int bft = DT_wind_msec_to_bft_indication(udp.windMperS);           // [0-8]
            int relDirCode = 0;
            if (TypeOfConsumptionModel == 3)
                relDirCode = convertWindRelDirToRelDirIndex(udp.windDir);
            else // if (TypeOfConsumptionModel == 4)
                relDirCode = convertWindRelDirToRelDirIndexConsType4(udp.windDir);

            int swell = DT_swell_meters_indication(udp.swell);               // [0-7] or [1-8]???
            string key = draftSel.ToString(nfi) + "_" + listOfDraftsSpeeds[draftIndex].listOfSpeeds[curspeedIndex].ToString(nfi) + "_" + bft + "_" + relDirCode + "_" + swell;
            try
            {
                consumptionPerMin = DecisionTree_Type_3[key] / convertFromDayToHour;
            }
            catch
            {

            }

            return consumptionPerMin;
        }

        public double CalculateConsumptionPerMinInterpolation(UnseenDataPoint udp)
        {
            double consumptionPerMin = 0;
            double convertFromDayToHour = 24;
            // Choose draft
            int draftIndex = -1;
            double draftSel = -1000;
            double minDraftDiff = 1000;
            for (int i = 0; i < listOfDraftsSpeeds.Count; i++)
            {
                double diff = Math.Abs(udp.draft - listOfDraftsSpeeds[i].draft);
                if (diff < minDraftDiff)
                {
                    minDraftDiff = diff;
                    draftSel = listOfDraftsSpeeds[i].draft;
                    draftIndex = i;
                }
            }

            //Choose speed
            double speedTemp = udp.stw;
            int curspeedIndex = -1;
            // 0 : Exact speed
            // -1: Interpolate with previous index
            // +1: Interpolate with next index
            int exactSpeedIndex = 0;
            if (udp.stw < listOfDraftsSpeeds[draftIndex].minSpeed)
                speedTemp = listOfDraftsSpeeds[draftIndex].minSpeed;
            else if (speedTemp > listOfDraftsSpeeds[draftIndex].maxSpeed)
                speedTemp = listOfDraftsSpeeds[draftIndex].maxSpeed;
            // Find where it is in the list of speeds
            double adjustSpeedGroup = 0.2;
            for (int i = 0; i < listOfDraftsSpeeds[draftIndex].listOfSpeeds.Count; i++)
            {
                curspeedIndex = i;
                if (speedTemp <= listOfDraftsSpeeds[draftIndex].listOfSpeeds[i] + adjustSpeedGroup)
                {
                    if (speedTemp == listOfDraftsSpeeds[draftIndex].listOfSpeeds[i])
                        exactSpeedIndex = 0;
                    else if (speedTemp > listOfDraftsSpeeds[draftIndex].listOfSpeeds[i])
                    {
                        if (i < listOfDraftsSpeeds[draftIndex].listOfSpeeds.Count - 1)
                            exactSpeedIndex = 1;
                        else
                            exactSpeedIndex = 0;
                    }
                    else // if (speedTemp < listOfDraftsSpeeds[draftIndex].listOfSpeeds[i])
                    {
                        if (i > 0)
                            exactSpeedIndex = -1;
                        else
                            exactSpeedIndex = 0;
                    }

                    break;
                }
            }
            int bft = DT_wind_msec_to_bft_indication(udp.windMperS);
            int relDirCode = 0;           // [0-8]
            if (TypeOfConsumptionModel == 3)
                relDirCode = convertWindRelDirToRelDirIndex(udp.windDir);
            else // if (TypeOfConsumptionModel == 4)
                relDirCode = convertWindRelDirToRelDirIndexConsType4(udp.windDir);
            int swell = DT_swell_meters_indication(udp.swell);               // [0-7] or [1-8]???
            if (exactSpeedIndex == 0)
            {
                string key = draftSel.ToString(nfi) + "_" + listOfDraftsSpeeds[draftIndex].listOfSpeeds[curspeedIndex].ToString(nfi) + "_" + bft + "_" + relDirCode + "_" + swell;
                try
                {
                    consumptionPerMin = DecisionTree_Type_3[key] / convertFromDayToHour;
                }
                catch
                {
                      int test;
                }
            }
            else
            {
                double lowSpeed;
                double highSpeed;
                if (exactSpeedIndex == 1)
                {
                    lowSpeed = listOfDraftsSpeeds[draftIndex].listOfSpeeds[curspeedIndex];
                    highSpeed = listOfDraftsSpeeds[draftIndex].listOfSpeeds[curspeedIndex + 1];
                }
                else //if (exactSpeedIndex == -1)
                {
                    lowSpeed = listOfDraftsSpeeds[draftIndex].listOfSpeeds[curspeedIndex - 1];
                    highSpeed = listOfDraftsSpeeds[draftIndex].listOfSpeeds[curspeedIndex];
                }
                // Speed: 12.7
                // Low: 12.5 => Cons(12.5) - dictionary
                // High: 13 => Cons(13) - dictionary
                // perc = (12.7^3  - 12.5^5) / (13^3 - 12.5^3)
                // consDiff = consHigh(13) - consLow(12.5)
                // cons(12.7) = consLow + perc * (cons(13) - cons(12.5))
                string keyLow = draftSel.ToString(nfi) + "_" + lowSpeed.ToString(nfi) + "_" + bft + "_" + relDirCode + "_" + swell;
                string keyHigh = draftSel.ToString(nfi) + "_" + highSpeed.ToString(nfi) + "_" + bft + "_" + relDirCode + "_" + swell;
                try
                {
                    
                    double calcAvgConsLow = DecisionTree_Type_3[keyLow] / convertFromDayToHour;
                    double calcAvgConsHigh = DecisionTree_Type_3[keyHigh] / convertFromDayToHour;
                    double difInCons = calcAvgConsHigh - calcAvgConsLow;
                    double percSpeedDif = (Math.Pow(speedTemp, 3) - Math.Pow(lowSpeed, 3)) / (Math.Pow(highSpeed, 3) - Math.Pow(lowSpeed, 3)); // Cubic interpolation
                    consumptionPerMin = calcAvgConsLow + difInCons * percSpeedDif; // Linear interpolation
                }
                catch
                {
                    int test;
                  
                }
            }

            return consumptionPerMin;
        }

        public double IdentifySpeed(UnseenDataPoint udp)
        {
            int draftIndex = -1;
            double minDraftDiff = 1000;
            for (int i = 0; i < listOfDraftsSpeeds.Count; i++)
            {
                double diff = Math.Abs(udp.draft - listOfDraftsSpeeds[i].draft);
                if (diff < minDraftDiff)
                {
                    minDraftDiff = diff;
                    draftIndex = i;
                }
            }

            //Choose speed
            double speedTemp = udp.stw;
            int curspeedIndex = -1;
            if (udp.stw < listOfDraftsSpeeds[draftIndex].minSpeed)
                speedTemp = listOfDraftsSpeeds[draftIndex].minSpeed;
            else if (speedTemp > listOfDraftsSpeeds[draftIndex].maxSpeed)
                speedTemp = listOfDraftsSpeeds[draftIndex].maxSpeed;
            // Find where it is in the list of speeds
            double adjustSpeedGroup = 0.25;
            for (int i = 0; i < listOfDraftsSpeeds[draftIndex].listOfSpeeds.Count; i++)
            {
                curspeedIndex = i;
                if (speedTemp <= listOfDraftsSpeeds[draftIndex].listOfSpeeds[i] + adjustSpeedGroup)
                    break;
            }
            return listOfDraftsSpeeds[draftIndex].listOfSpeeds[curspeedIndex];
        }

        public int DT_wind_msec_to_bft_indication(double windSpeedMperS)
        {
            int beaufort;

            if (TypeOfConsumptionModel != 4)
            {
                if (windSpeedMperS < 1.6)
                    beaufort = 1;
                else if (windSpeedMperS < 3.4)
                    beaufort = 2;
                else if (windSpeedMperS < 5.5)
                    beaufort = 3;
                else if (windSpeedMperS < 8)
                    beaufort = 4;
                else if (windSpeedMperS < 10.8)
                    beaufort = 5;
                else if (windSpeedMperS < 13.9)
                    beaufort = 6;
                else if (windSpeedMperS < 17.2)
                    beaufort = 7;
                else
                    beaufort = 8;
            }
            else //  if (TypeOfConsumptionModel == 4)
            {
                if (windSpeedMperS < 0.5)
                    beaufort = 0;
                else if (windSpeedMperS < 1.6)
                    beaufort = 1;
                else if (windSpeedMperS < 3.4)
                    beaufort = 2;
                else if (windSpeedMperS < 5.5)
                    beaufort = 3;
                else if (windSpeedMperS < 8)
                    beaufort = 4;
                else if (windSpeedMperS < 10.8)
                    beaufort = 5;
                else if (windSpeedMperS < 13.9)
                    beaufort = 6;
                else if (windSpeedMperS < 17.2)
                    beaufort = 7;
                else if (windSpeedMperS < 20.7)
                    beaufort = 8;
                else 
                    beaufort = 9;
            }

            return beaufort;
        }
        public int DT_swell_meters_indication(double swellSWH)
        {
            int swellInd = 0;
            if (swellSWH > 8)
                swellInd = 8;

            if (TypeOfConsumptionModel != 4)
            {
              
    
                if (swellSWH < 1)
                    swellInd = 1;

                swellInd = Convert.ToInt32(Math.Round(swellSWH));
            }
            else // if (TypeOfConsumptionModel == 4)
            {
                if (swellSWH < 0)
                    swellInd = 0;
                else    
                    swellInd = Convert.ToInt32(Math.Round(swellSWH * 2));
            }

            return swellInd;
        }

        public int convertWindRelDirToRelDirIndex(double _weatherRelDir)
        {
            int relDirCode;
            if (_weatherRelDir <= 180)
            {
                if (_weatherRelDir <= 22.5)
                    relDirCode = 0;
                else if (((_weatherRelDir > 22.5) && (_weatherRelDir <= 67.5)))
                    relDirCode = 1;
                else if (((_weatherRelDir > 67.5) && (_weatherRelDir <= 112.5)))
                    relDirCode = 2;
                else if (((_weatherRelDir > 112.5) && (_weatherRelDir <= 157.5)))
                    relDirCode = 3;
                else // if (_weatherRelDir >  157.5)
                    relDirCode = 4;
            }
            else
            {
                if (_weatherRelDir >= 337.5)
                    relDirCode = 0;
                else if (((_weatherRelDir < 337.5) && (_weatherRelDir >= 292.5)))
                    relDirCode = 1;
                else if (((_weatherRelDir < 292.5) && (_weatherRelDir >= 247.5)))
                    relDirCode = 2;
                else if (((_weatherRelDir < 247.5) && (_weatherRelDir >= 202.5)))
                    relDirCode = 3;
                else // if (_weatherRelDir <  202.5)
                    relDirCode = 4;
            }

            return relDirCode;
        }

        public int convertWindRelDirToRelDirIndexConsType4(double _weatherRelDir)
        {
            int relDirCode;
            if (_weatherRelDir < 90)
            {
                if (_weatherRelDir < 15)
                    relDirCode = 0;
                else if (((_weatherRelDir >= 15) && (_weatherRelDir < 30)))
                    relDirCode = 1;
                else if (((_weatherRelDir >= 30) && (_weatherRelDir < 45)))
                    relDirCode = 2;
                else if (((_weatherRelDir >= 45) && (_weatherRelDir < 60)))
                    relDirCode = 3;
                else if (((_weatherRelDir >= 60) && (_weatherRelDir < 75)))
                    relDirCode = 4;
                else // if (_weatherRelDir >=  75)  && (_weatherRelDir < 90)
                    relDirCode = 5;
            }
            else if (_weatherRelDir < 180)
            {
                if (_weatherRelDir < 105)
                    relDirCode = 6;
                else if (((_weatherRelDir >= 105) && (_weatherRelDir < 120)))
                    relDirCode = 7;
                else if (((_weatherRelDir >= 120) && (_weatherRelDir < 135)))
                    relDirCode = 8;
                else if (((_weatherRelDir >= 135) && (_weatherRelDir < 150)))
                    relDirCode = 9;
                else if (((_weatherRelDir >= 150) && (_weatherRelDir < 165)))
                    relDirCode = 10;
                else // if (_weatherRelDir >= 165)  && (_weatherRelDir < 180)
                    relDirCode = 11;
            }
            else if (_weatherRelDir < 270)
            {
                if (_weatherRelDir < 195)
                    relDirCode = 11;
                else if (((_weatherRelDir >= 195) && (_weatherRelDir < 210)))
                    relDirCode = 10;
                else if (((_weatherRelDir >= 210) && (_weatherRelDir < 225)))
                    relDirCode = 9;
                else if (((_weatherRelDir >= 225) && (_weatherRelDir < 240)))
                    relDirCode = 8;
                else if (((_weatherRelDir >= 240) && (_weatherRelDir < 255)))
                    relDirCode = 7;
                else // if (_weatherRelDir >= 255)  && (_weatherRelDir < 270)
                    relDirCode = 6;
            }
            else // if (_weatherRelDir < 360)
            {
                if (_weatherRelDir < 285)
                    relDirCode = 5;
                else if (((_weatherRelDir >= 285) && (_weatherRelDir < 300)))
                    relDirCode = 4;
                else if (((_weatherRelDir >= 300) && (_weatherRelDir < 315)))
                    relDirCode = 3;
                else if (((_weatherRelDir >= 315) && (_weatherRelDir < 330)))
                    relDirCode = 2;
                else if (((_weatherRelDir >= 330) && (_weatherRelDir < 345)))
                    relDirCode = 1;
                else // if (_weatherRelDir >= 345)  && (_weatherRelDir <= 360)
                    relDirCode = 0;
            }

            return relDirCode;
        }
    }

    public class BasicProfileSettingsPerDraft
    {
        public double draft { set; get; }
        public List<ConsProfileTable_3> cp { set; get; }
        public List<double> listOfSpeeds { set; get; }
        public double minSpeed { set; get; }
        public double maxSpeed { set; get; }

        public BasicProfileSettingsPerDraft(double _draft)
        {
            draft = _draft;
            listOfSpeeds = new List<double>();
        }
    }

    public class UnseenDataPoint
    {
        public double draft { set; get; }
        public double stw { set; get; }
        public double windMperS { set; get; }
        public double windDir { set; get; } // Relative dir
        public double swell { set; get; }
        public double cons { set; get; }  // MT/min
    }

    public class UnseenData
    {

        public string vessel_code { set; get; }
        public string vessel_name { set; get; }
        public string dateCreated { set; get; }
        public List<UnseenDataPoint> data { set; get; }
    }

    public class UnseenDataSet
    {   
        public UnseenData ConsumptionProfile_Dataset { set; get; }
    }

    public class ConsumptionProfileFull_3
    {
        public ConsumptionProfile_3 ConsumptionProfile { set; get; }
    }

    public class ConsumptionProfile_3
    {
        public string vessel_code { set; get; }
        public string vessel_name { set; get; }
        public string dateCreated { set; get; }
        public List<ConsProfileTable_3> consProfile { set; get; }
    }

    public class ConsProfileTable_3
    {
        public double draft { set; get; }
        public double speed { set; get; }
        public List<ConsProfileItem_3> cells { set; get; }
    }

    public class ConsProfileItem_3
    {
        public int windBFT { set; get; }
        public double windDir { set; get; }
        public double swell { set; get; }
        public double cons { set; get; }
    }

    public class ConnStr
    {
        
        public string Down { set; get; }
        public string Prod { set; get; }
        public string JSON { set; get; }
        public string DownData { set; get; }
        public string ProdData { set; get; }
        public string ServerNameDown { set; get; }
        public string DownDownloadDB { set; get; }
        public string DownRoutingDB { set; get; }
        public string ProdDownloadDB { set; get; }
        public string ProdRoutingDB { set; get; }
    }
}
