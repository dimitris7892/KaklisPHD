using System;
using System.Collections.Generic;
using System.Linq;
using System.Configuration;
using System.Threading.Tasks;
using System.Reflection;
using System.IO;
using System.Text;
using ConsModelComparison;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

namespace ConsModelComparison_webapi.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class CompareConsumptionModelsController : ControllerBase
    {
        private readonly ILogger<CompareConsumptionModelsController> _logger;
        private CompareConsumptionModels _service;


        public CompareConsumptionModelsController(ILogger<CompareConsumptionModelsController> logger, CompareConsumptionModels service)
        {
            _logger = logger;
            _service = service;
        }

        [HttpGet]
        public void Get(bool interpolation, string vessel)
        {   
            string pathUnseenData = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + @"/TestData_"+vessel+"_.json";
            string pathConsProfileNeural = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + @"/consProfile_"+vessel+"_NeuralDT.json";
            string pathWrite = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + @"/compareSpeed";

            _service.CompareModels("connStr", pathUnseenData, pathConsProfileNeural, pathWrite, interpolation);
            //return response;
            // return Enumerable.Range(0, response.Count).Select(index => new CompareConsModels
            // {   
            //     Speed = response[index].speed,
            //     Size = response[index].sizeOfDataset,
            //     NeuralDT_Acc = response[index].percDiffModel1,
            //     StatsDT_Acc = response[index].percDiffModel2,
            //     Actual_Avg = response[index].avgActual * 24,
            //     NeuralDT_Avg = response[index].avgModel1 * 24,
            //     Stats_Avg = response[index].avgModel2 * 24,
            //     NeuralDT_RMSE = response[index].rmseModel1,
            //     Stats_RMSE = response[index].rmseModel2,
            // })
            // .ToArray();

        }
    }
}
