using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using UpdateVesselProfile;
using System.Configuration;
//using System.IO.FileSystem;
using System.Reflection;
using System.Text;


namespace UpdateVesselProfile_webapi.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class UpdateVesselProfileController : ControllerBase
    {
       

        private readonly ILogger<UpdateVesselProfileController> _logger;
        private ManageVesselProfile _service;
        public UpdateVesselProfileController(ILogger<UpdateVesselProfileController> logger, ManageVesselProfile service)
        {
            _logger = logger;
            _service = service;
        }

       [HttpGet]
        public void Get(string updateServer)
        {   
            string weatherRoutingDB_WS = ConfigurationManager.AppSettings["connectionStringWR_WS"];
            string weatherReportDB_WS = ConfigurationManager.AppSettings["connectionStringWRep_WS"];
            string weatherRoutingDB_MS = ConfigurationManager.AppSettings["connectionStringWR_MS"];
            string weatherReportDB_MS = ConfigurationManager.AppSettings["connectionStringWRep_MS"];

            int cellsPerTable = Convert.ToInt32(ConfigurationManager.AppSettings["cellsPerTable"]);

            string consProfilePath = System.IO.Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + "/consProfile.json";
            string consProfileStr = System.IO.File.ReadAllText(consProfilePath);


            //_service.CompareModels("connStr", pathUnseenData, pathConsProfileNeural, pathWrite, interpolation);

            //ManageVesselProfile mvp = new ManageVesselProfile();
            if (updateServer == "WeatherServer")
                _service.InsertOrUpdateProfile(consProfileStr, cellsPerTable, weatherRoutingDB_WS, weatherReportDB_WS, "WeatherServer");
             else
                _service.InsertOrUpdateProfile(consProfileStr, cellsPerTable, weatherRoutingDB_MS, weatherReportDB_MS, "MapServer");
            //return response;
        }
    }
}
