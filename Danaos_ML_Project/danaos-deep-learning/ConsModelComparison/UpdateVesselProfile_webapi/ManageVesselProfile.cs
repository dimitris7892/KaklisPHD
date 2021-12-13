using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Microsoft.Extensions.Configuration;

namespace UpdateVesselProfile
{
    
     public class ManageVesselProfile
    {

        private static IConfiguration _config;
        
        public ManageVesselProfile(IConfiguration config)
        {
            
            _config = config;
        }
        public void InsertOrUpdateProfile(string consProfileStr, int cellsPerTable, string weatherRoutingDBstr, string weatherReportDBstr, string DBserverName)
        {
            ConsumptionProfileFull_3 cp = JsonConvert.DeserializeObject<ConsumptionProfileFull_3>(consProfileStr);
            bool validModel = CheckValidityOfModel(cp, cellsPerTable);
            if (validModel)
            {
                int profType = 3;
                // ------------ NEURAL TESTING --------------------
                //cp.ConsumptionProfile.vessel_code = "9484948-N";
                //profType = 4;
                // ------------------------------------------------
                bool foundInReportDB = CheckWeatherReportDB(cp.ConsumptionProfile.vessel_code, weatherReportDBstr);
                bool continueInDB = true;
                if (foundInReportDB)
                {
                    Console.WriteLine("The vessel already has a profile in " + DBserverName + ".");
                    bool answer = false;
                    while (!answer)
                    {
                        Console.WriteLine("Do you want to replace the existing one (Y/N)?");
                        string answerStr = "Y";
                        
                        if ((answerStr == "Y") || (answerStr == "y"))
                        {
                            continueInDB = true;
                            answer = true;
                        }
                        else if ((answerStr == "N") || (answerStr == "n"))
                        {
                            continueInDB = false;
                            answer = true;
                        }
                    }
                }
                if (continueInDB)
                {
                    WeatherReportModel wrm = new WeatherReportModel(cp.ConsumptionProfile.vessel_code, cp.ConsumptionProfile.vessel_name, profType);
                    if (foundInReportDB) // We update it
                    {
                        Console.WriteLine("Updating Vessel record in WeatherReportDB - " + DBserverName + ".");
                        UpdateWeatherReportDB(wrm, weatherReportDBstr);
                    }
                    else // We insert it
                    {
                        Console.WriteLine("Inserting Vessel record in WeatherReportDB - " + DBserverName + ".");
                        InsertWeatherReportDB(wrm, weatherReportDBstr);
                    }

                    bool foundInRoutingDB = CheckWeatherRoutingDB(cp.ConsumptionProfile.vessel_code, weatherRoutingDBstr);
                    if (foundInRoutingDB) // We update it
                    {
                        Console.WriteLine("Updating Vessel profile in WeatherRoutingDB - " + DBserverName + ".");
                        //UpdateWeatherRoutingDB("9484948-N", consProfileStr, weatherRoutingDBstr);
                        UpdateWeatherRoutingDB(cp.ConsumptionProfile.vessel_code, consProfileStr, weatherRoutingDBstr);
                    }
                    else // We insert it
                    {
                        Console.WriteLine("Inserting Vessel profile in WeatherRoutingDB - " + DBserverName + ".");
                        InsertWeatherRoutingDB(cp.ConsumptionProfile.vessel_code, consProfileStr, weatherRoutingDBstr);
                    }
                    Console.WriteLine("The Vessel profile has been inserted in in WeatherRoutingDB - " + DBserverName + ".");
                }
            }
            else
            {
                Console.WriteLine("The model is not valid.");
                Console.WriteLine("It has not been inserted/updated in " + DBserverName + ".");
            }
            Console.WriteLine("Press 'Enter' to continue.");
            Console.ReadLine();
        }

        public bool CheckValidityOfModel(ConsumptionProfileFull_3 cp, int cellsPerTable)
        {
            bool valid = true;

            //for (int i = 0; i < cp.ConsumptionProfile.consProfile.Count; i++)
            //    if (cp.ConsumptionProfile.consProfile[i].cells.Count != cellsPerTable)
            //    {
            //        Console.WriteLine("Validity check " + i.ToString() + ": The table does not have " + cellsPerTable.ToString() + " cells.");
            //        valid = false;
            //    }

            //if (cp.ConsumptionProfile.vessel_code.Length != 7)
            //{
            //    Console.WriteLine("Validity check - Vessel Code: It is not 7 characters (IMO number)");
            //    valid = false;
            //}

            //if (!IsDigitsOnly(cp.ConsumptionProfile.vessel_code))
            //{
            //    Console.WriteLine("Validity check - Vessel Code: It is not an IMO number (it does not contain 7 digits)");
            //    valid = false;
            //}

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

        public bool CheckWeatherReportDB(string vesselCode, string weatherreportConnStr)
        {
            bool found = false;

            using (SqlConnection sqlConnection = new SqlConnection(weatherreportConnStr))
            {
                sqlConnection.Open();
                using (SqlCommand cmd = new SqlCommand())
                {
                    cmd.CommandType = CommandType.Text;
                    cmd.Connection = sqlConnection;
                    cmd.CommandText = "SELECT count(*) FROM BasicInfo WHERE vesselCode = '" + vesselCode + "'";
                    int selectCount = (int)cmd.ExecuteScalar();
                    if (selectCount > 0)
                        found = true;
                }

                sqlConnection.Close();
            }

            return found;
        }

        public void UpdateWeatherReportDB(WeatherReportModel wrm, string weatherreportConnStr)
        {
            using (SqlConnection sqlConnection1 = new SqlConnection(weatherreportConnStr))
            {
                sqlConnection1.Open();
                SqlCommand cmd = new SqlCommand();
                cmd.CommandType = CommandType.Text;
                cmd.Connection = sqlConnection1;
                cmd.CommandText = "UPDATE BasicInfo SET vesselName = '" + wrm.vesselName + "', profileType = " + wrm.profileType + " WHERE vesselCode = '" + wrm.vesselCode + "'";
                cmd.ExecuteNonQuery();
                sqlConnection1.Close();
            }
        }

        public void InsertWeatherReportDB(WeatherReportModel wrm, string weatherreportConnStr)
        {
            using (SqlConnection sqlConnection1 = new SqlConnection(weatherreportConnStr))
            {
                int map_0_360_Int = 0;
                if (wrm.map_0_360)
                    map_0_360_Int = 1;
                int enabledInt = 0;
                if (wrm.enabled)
                    enabledInt = 1;
                string dt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
                sqlConnection1.Open();
                SqlCommand cmd = new SqlCommand();
                cmd.CommandType = CommandType.Text;
                cmd.Connection = sqlConnection1;
                cmd.CommandText = "INSERT INTO BasicInfo VALUES ('" + wrm.vesselCode + "','" + wrm.vesselName + "'," + wrm.minLon + "," + wrm.maxLon + "," + wrm.minLat + "," + wrm.maxLat + "," +
                                                                    map_0_360_Int + "," + enabledInt + ",'" + wrm.MC + "'," + wrm.profileType + ", '" + dt + "')";
                cmd.ExecuteNonQuery();
                sqlConnection1.Close();
            }
        }

        public bool CheckWeatherRoutingDB(string vesselCode, string weatherroutingConnStr)
        {
            bool found = false;

            using (SqlConnection sqlConnection = new SqlConnection(weatherroutingConnStr))
            {
                sqlConnection.Open();
                using (SqlCommand cmd = new SqlCommand())
                {
                    cmd.CommandType = CommandType.Text;
                    cmd.Connection = sqlConnection;
                    cmd.CommandText = "SELECT count(*) FROM ConsProfileNew WHERE vesselCode = '" + vesselCode + "'";
                    int selectCount = (int)cmd.ExecuteScalar();
                    if (selectCount > 0)
                        found = true;
                }

                sqlConnection.Close();
            }

            return found;
        }

        public void UpdateWeatherRoutingDB(string vesselCode, string profileStr, string weatherroutingConnStr)
        {
            using (SqlConnection sqlConnection1 = new SqlConnection(weatherroutingConnStr))
            {
                sqlConnection1.Open();
                SqlCommand cmd = new SqlCommand();
                cmd.CommandType = CommandType.Text;
                cmd.Connection = sqlConnection1;
                cmd.CommandText = "UPDATE ConsProfileNew SET profile = '" + profileStr + "' WHERE vesselCode = '" + vesselCode + "'";
                cmd.ExecuteNonQuery();
                sqlConnection1.Close();
            }
        }

        public void InsertWeatherRoutingDB(string vesselCode, string profileStr, string weatherroutingConnStr)
        {
            using (SqlConnection sqlConnection1 = new SqlConnection(weatherroutingConnStr))
            {
                sqlConnection1.Open();
                SqlCommand cmd = new SqlCommand();
                cmd.CommandType = CommandType.Text;
                cmd.Connection = sqlConnection1;
                cmd.CommandText = "INSERT INTO ConsProfileNew VALUES ('" + vesselCode + "','" + profileStr + "')";
                cmd.ExecuteNonQuery();
                sqlConnection1.Close();
            }
        }
    }

    public class WeatherReportModel
    {
        public string vesselCode { set; get; }
        public string vesselName { set; get; }
        public int profileType { set; get; }
        // --- These were utilized in the previous (email) version of the service ----
        public double minLon { set; get; }
        public double maxLon { set; get; }
        public double minLat { set; get; }
        public double maxLat { set; get; }
        public bool map_0_360 { set; get; }
        public bool enabled { set; get; }
        public string MC { set; get; }
        // --------------------------------------------------------------------------

        public WeatherReportModel(string vesselCode, string vesselName, int profType)
        {
            this.vesselCode = vesselCode;
            this.vesselName = vesselName;
            this.profileType = profType;
            minLon = -180;
            maxLon = 180;
            minLat = -85;
            maxLat = 85;
            map_0_360 = false;
            enabled = false;
            MC = "";
        }
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
}
