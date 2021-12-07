using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using RouteBreakGen;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Configuration;
using System.Threading.Tasks;
using System.Reflection;
using System.IO;
using System.Text;
using Newtonsoft.Json;

namespace RouteBreakGen_webapi.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class BreakRouteController : ControllerBase
    {
       

        private readonly ILogger<BreakRouteController> _logger;
        private BreakRoute _service;

        public BreakRouteController(ILogger<BreakRouteController> logger, BreakRoute service)
        {
            _logger = logger;
            _service = service;
        }

        [HttpGet]
        public ActionResult<IEnumerable<WaypointNew>> Get(string vsLeg)
        {

            string pathOnBoardOutput = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) + @"/response_"+vsLeg+".json";
            string onBoardOutput =  System.IO.File.ReadAllText(pathOnBoardOutput);

            onBoardOutput = Newtonsoft.Json.JsonConvert.DeserializeObject(onBoardOutput).ToString();
            OnboardOutput oo = Newtonsoft.Json.JsonConvert.DeserializeObject<OnboardOutput>(onBoardOutput);

          
             _service.AddTimeBreakPointsSpeed(oo, 1);
            //return oo;
            return Enumerable.Range(0, oo.Path.Count).Select(index => new WaypointNew
            {   
                Lon = oo.Path[index].Lon,
                Lat = oo.Path[index].Lat,
                PCode = oo.Path[index].PCode,
                PName = oo.Path[index].PName,
                Speed = oo.Path[index].Speed ,
                GC = oo.Path[index].GC,
                Locked = oo.Path[index].Locked,
                WaitHr = oo.Path[index].WaitHr,
                
            })
            .ToArray();
        }
    }
}
