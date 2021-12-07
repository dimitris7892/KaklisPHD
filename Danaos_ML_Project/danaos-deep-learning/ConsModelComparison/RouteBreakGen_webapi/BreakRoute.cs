using System;
using System.Collections.Generic;
using Microsoft.Extensions.Configuration;

namespace RouteBreakGen
{
    public class BreakRoute
    {   

        private static IConfiguration _config;
        
        public BreakRoute(IConfiguration config)
        {
            
            _config = config;
        }

        public void AddTimeBreakPointsSpeed(OnboardOutput oo, double timeBreakStep)
        {
            Calculation calc = new Calculation();
            List<WaypointNew> newPath = new List<WaypointNew>();
            if (oo.Path.Count > 0)
            {
                // Handle the first point
                double prevBasiclon = oo.Path[0].Lon;
                double prevBasiclat = oo.Path[0].Lat;
                newPath.Add(oo.Path[0]);
                // Handle the rest of the points
                for (int i = 1; i < oo.Path.Count; i++)
                {
                    WaypointNew curBasicPoint = oo.Path[i];
                    double curBasiclon = oo.Path[i].Lon;
                    double curBasiclat = oo.Path[i].Lat;
                    double curSpeedOverGround = oo.Path[i].Speed;
                    double distFromPrev = calc.CalcDistGC(prevBasiclon, prevBasiclat, curBasiclon, curBasiclat);
                    double distanceBound = timeBreakStep * curSpeedOverGround;
                    // If the distance is already smaller, no need to add any new "T" points
                    if (distFromPrev <= distanceBound)
                    {
                        newPath.Add(oo.Path[i]);
                        prevBasiclon = curBasiclon;
                        prevBasiclat = curBasiclat;
                    }
                    else // If the distance is larger, we add the corresponding "T" points
                    {
                        double ratio = distanceBound / distFromPrev;  // This is smaller than 1
                        int times = Convert.ToInt32(1.0 / ratio) - 1;     // We find the integer part of the division to know how many points we will be creating
                        double distInDeg = calc.CalcDistInDeg(prevBasiclon, prevBasiclat, curBasiclon, curBasiclat);
                        // In this case we know we need to add at least one point
                        for (int j = 1; j <= times; j++)
                        {
                            double curDistDeg = j * ratio * distInDeg;
                            List<double> newPointT = calc.FindPointOnLineGivenDistance(prevBasiclon, prevBasiclat, curBasiclon, curBasiclat, curDistDeg);
                            WaypointNew newTPoint = new WaypointNew();
                            newTPoint.Lon = newPointT[0];
                            newTPoint.Lat = newPointT[1];
                            newTPoint.Speed = curSpeedOverGround;
                            newPath.Add(newTPoint);
                        }
                        newPath.Add(oo.Path[i]);
                        // We need to recalculate the distance from the previous point
                        prevBasiclon = curBasiclon;
                        prevBasiclat = curBasiclat;
                    }
                }

                oo.Path = newPath;
            }
        }
    }

    public class OnboardOutput
    {
        public bool FoundSol { set; get; }
        public bool Crushed { set; get; }
        public string ErrorMessage { set; get; }
        public List<WaypointNew> Path = new List<WaypointNew>();
        public List<RouteMetadata> Metadata = new List<RouteMetadata>();

        public OnboardOutput()
        {
            Crushed = false;
            FoundSol = true;
            ErrorMessage = "";
        }

        public void AddWaypointNew(double lon, double lat, string pcode, string pname, double speed, bool gc, bool locked, double waitHr)
        {
            WaypointNew wn = new WaypointNew();
            wn.Lon = lon;
            wn.Lat = lat;
            wn.PCode = pcode;
            wn.PName = pname;
            wn.Speed = speed;
            wn.GC = gc;
            wn.Locked = locked;
            wn.WaitHr = waitHr;
            Path.Add(wn);
        }

        public void UpdateVoyageCoordinatesAccuracy()
        {
            // Round points
            for (int x = 0; x < Path.Count; x++)
            {
                Path[x].Lon = Math.Round(Path[x].Lon, 6);
                Path[x].Lat = Math.Round(Path[x].Lat, 6);
            }
            // Check if points with same coordinates have been created
            int k = 1;
            while (k < Path.Count)
            {
                if ((Path[k - 1].Lon == Path[k].Lon) && (Path[k - 1].Lat == Path[k].Lat))
                {
                    double hoursBeforeDepart = 0;
                    if ((Path[k - 1].WaitHr > 0) || (Path[k].WaitHr > 0))
                        hoursBeforeDepart = Path[k - 1].WaitHr + Path[k].WaitHr;
                    double speedFromPrev = Path[k - 1].Speed;
                    if ((Path[k - 1].Speed != Path[k].Speed) && (k == 1))
                        speedFromPrev = Path[k].Speed;

                    if (k == 1)
                        Path.RemoveAt(k); // Keep the first point
                    else if (k == Path.Count - 1)
                        Path.RemoveAt(k - 1); // Keep the last point
                    else
                    {
                        if ((Path[k - 1].PCode != "") || (Path[k - 1].PName != ""))
                            Path.RemoveAt(k); // Keep the routing point
                        else if ((Path[k].PCode != "") || (Path[k].PName != ""))
                            Path.RemoveAt(k - 1); // Keep the routing point
                        else
                        {
                            if ((Path[k - 1].Locked) && (!Path[k].Locked))
                                Path.RemoveAt(k); // Keep the fixed point
                            else if ((!Path[k - 1].Locked) && (Path[k].Locked))
                                Path.RemoveAt(k - 1); // Keep the fixed point
                            else
                            {
                                Path.RemoveAt(k); // Keep randomly a point
                            }
                        }
                    }
                    Path[k - 1].WaitHr = hoursBeforeDepart;
                    Path[k - 1].Speed = speedFromPrev;
                }
                else
                    k++;
            }
        }
    }

    public class WaypointNew
    {
        public double Lon { set; get; }
        public double Lat { set; get; }
        public string PCode { set; get; }
        public string PName { set; get; }
        public double Speed { set; get; }
        public bool GC { set; get; }
        public bool Locked { set; get; }
        public double WaitHr { set; get; }

        public WaypointNew(double Lon, double Lat, string PCode, string PName, double Speed, bool GC, bool Locked, double WaitHr)
        {
            this.Lon = Lon;
            this.Lat = Lat;
            this.PCode = PCode;
            this.PName = PName;
            this.Speed = Speed;
            this.GC = GC;
            this.Locked = Locked;
            this.WaitHr = WaitHr;
        }

        public WaypointNew()
        {
            Lon = 0;
            Lat = 0;
            PCode = "";
            PName = "";
            Speed = 12;
            GC = false;
            Locked = false;
            WaitHr = 0;
        }
    }

    public class RouteMetadata
    {
        public int From { set; get; }
        public int To { set; get; }
        public string Name { set; get; }
        public string Type { set; get; }

        public RouteMetadata(int from, int to, string name, string type)
        {
            From = from;
            To = to;
            Name = name;
            Type = type;
        }
    }

    public class Calculation
    {
        // --- BASIC MATH CALCULATIONS ---

        public LineSlope CalcSlope(double lon1, double lat1, double lon2, double lat2)
        {
            LineSlope lineSlope = new LineSlope();
            if (lon1 == lon2)
            {
                lineSlope.Slope = 0;
                lineSlope.SlopeType = "infint";
            }
            else
            {
                lineSlope.Slope = Math.Round((lat1 - lat2) / (lon1 - lon2), 12);
                lineSlope.SlopeType = "normal";
            }

            return lineSlope;
        }

        public LineSlope CalcSlopeMOD(double lon1, double lat1, double lon2, double lat2, double mod)
        {
            LineSlope lineSlope = new LineSlope();
            if (lon1 == lon2)
            {
                lineSlope.SlopeType = "infint";
                lineSlope.Slope = 0;
            }
            else
            {
                lineSlope.SlopeType = "normal";
                lineSlope.Slope = Math.Round((lat1 - lat2) / (Mod_ChangeNum(lon1, mod) - Mod_ChangeNum(lon2, mod)), 12);
            }

            return lineSlope;
        }

        //Factor to convert decimal degrees to radians
        //double DEG2RAD = 0.01745329252;
        public double DegreeToRadian(double angle)
        {
            return Math.PI * angle / 180.0;
        }

        //Factor to convert decimal degrees to radians
        //double RAD2DEG = 57.29577951308;
        public double RadianToDegree(double radian)
        {
            return radian * 180 / Math.PI;
        }

        public double CalcSinAngle(double lon1, double lat1, double lon2, double lat2)
        {
            double sinAngle;
            double oppos = lat2 - lat1;
            double adjac = lon2 - lon1;
            double hypot = Math.Sqrt(Math.Pow(oppos, 2) + Math.Pow(adjac, 2));

            if ((lon1 == lon2) && (lat1 == lat2))
                sinAngle = 0; // this case never happens
            else if (lon1 == lon2)
                sinAngle = 1;
            else if (lat1 == lat2)
                sinAngle = 0;
            else
            {
                sinAngle = (oppos / hypot);
                //   if (((endPoint.getX() > startPoint.getX()) && (startPoint.getY() > endPoint.getY())) || ((endPoint.getX() < startPoint.getX()) && (startPoint.getY() < endPoint.getY())))
                //       sinAngle = -1 * sinAngle;
            }

            return sinAngle;
        }

        public double CalcSinAngleMOD(double lon1, double lat1, double lon2, double lat2, double mod)
        {
            double sinAngle;
            double oppos = lat2 - lat1;
            double adjac = Mod_ChangeNum(lon2, mod) - Mod_ChangeNum(lon1, mod);
            double hypot = Math.Sqrt(Math.Pow(oppos, 2) + Math.Pow(adjac, 2));

            if ((lon1 == lon2) && (lat1 == lat2))
                sinAngle = 0; // this case never happens
            else if (lon1 == lon2)
                sinAngle = 1;
            else if (lat1 == lat2)
                sinAngle = 0;
            else
            {
                sinAngle = (oppos / hypot);
                //   if (((endPoint.getX() > startPoint.getX()) && (startPoint.getY() > endPoint.getY())) || ((endPoint.getX() < startPoint.getX()) && (startPoint.getY() < endPoint.getY())))
                //       sinAngle = -1 * sinAngle;
            }

            return sinAngle;
        }

        public double CalcCosAngle(double lon1, double lat1, double lon2, double lat2)
        {
            double cosAngle;
            double oppos = lat2 - lat1;
            double adjac = lon2 - lon1;
            double hypot = Math.Sqrt(Math.Pow(oppos, 2) + Math.Pow(adjac, 2));

            if ((lon1 == lon2) && (lat1 == lat2))
                cosAngle = 0; // this case never happens
            else if (lon1 == lon2)
                cosAngle = 0;
            else if (lat1 == lat2)
                cosAngle = 1;
            else
            {
                cosAngle = (adjac / hypot);
                //      if (((endPoint.getX() > startPoint.getX()) && (startPoint.getY() > endPoint.getY())) || ((endPoint.getX() < startPoint.getX()) && (startPoint.getY() < endPoint.getY())))
                //          cosAngle = -1 * cosAngle;
            }

            return cosAngle;
        }

        public double CalcCosAngleMOD(double lon1, double lat1, double lon2, double lat2, double mod)
        {
            double cosAngle;
            double oppos = lat2 - lat1;
            double adjac = Mod_ChangeNum(lon2, mod) - Mod_ChangeNum(lon1, mod);
            double hypot = Math.Sqrt(Math.Pow(oppos, 2) + Math.Pow(adjac, 2));

            if ((lon1 == lon2) && (lat1 == lat2))
                cosAngle = 0; // this case never happens
            else if (lon1 == lon2)
                cosAngle = 0;
            else if (lat1 == lat2)
                cosAngle = 1;
            else
            {
                cosAngle = (adjac / hypot);
                //      if (((endPoint.getX() > startPoint.getX()) && (startPoint.getY() > endPoint.getY())) || ((endPoint.getX() < startPoint.getX()) && (startPoint.getY() < endPoint.getY())))
                //          cosAngle = -1 * cosAngle;
            }

            return cosAngle;
        }

        public double Mod_ChangeNum(double lon, double changeNum)
        {
            if (lon < changeNum)
                lon = 360 + lon;

            return lon;
        }

        public double Mod_ChangeBack(double lon)
        {
            if (lon > 180)
                lon = lon - 360;

            return lon;
        }

        public int DecideSideOfLine(double Ax, double Ay, double Bx, double By, double X, double Y)
        {
            return Math.Sign(((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax)));
        }

        public double CalculateDistOfPointFromLine(double lp1X, double lp1Y, double lp2X, double lp2Y, double pX, double pY)
        {
            return Math.Abs(((lp2Y - lp1Y) * pX - (lp2X - lp1X) * pY + lp2X * lp1Y - lp2Y * lp1X)) / (Math.Sqrt(Math.Pow(lp2Y - lp1Y, 2) + Math.Pow(lp2X - lp1X, 2)));
        }

        public List<double> ProjectPointOnLine(double linePoint1X, double linePoint1Y, double linePoint2X, double linePoint2Y, double pointToProjectX, double pointToProjectY)
        {
            List<double> pointOnLine = new List<double>();

            if (linePoint1X == linePoint2X)
            {
                pointOnLine.Add(linePoint1X);
                pointOnLine.Add(pointToProjectY);
            }
            else
            {
                double a1 = linePoint2X - linePoint1X;
                double a2 = linePoint2Y - linePoint1Y;
                double a3 = linePoint1Y - linePoint2Y;
                double a4 = linePoint2X - linePoint1X;
                double b1 = pointToProjectX * (linePoint2X - linePoint1X) + pointToProjectY * (linePoint2Y - linePoint1Y);
                double b2 = linePoint1Y * (linePoint2X - linePoint1X) - linePoint1X * (linePoint2Y - linePoint1Y);

                double projX = (a4 * b1 - a2 * b2) / (a1 * a4 - a2 * a3);
                double projY = (b2 - a3 * projX) / a4;

                pointOnLine.Add(Math.Round(projX, 8));
                pointOnLine.Add(Math.Round(projY, 8));
            }

            return pointOnLine;
        }

        public List<double> ProjectPointOnLineIdentifyChangeDir(double linePoint1X, double linePoint1Y, double linePoint2X, double linePoint2Y, double pointToProjectX, double pointToProjectY)
        {
            List<double> pointOnLine = new List<double>();

            if (IdentifyChangeDirection_Map(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y)) // Handles the stapling of the earth.
            {
                double mod = FindNewMod(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y);
                linePoint1X = Mod_ChangeNum(linePoint1X, mod);
                linePoint2X = Mod_ChangeNum(linePoint2X, mod);
                pointToProjectX = Mod_ChangeNum(pointToProjectX, mod);
            }

            if (linePoint1X == linePoint2X)
            {
                pointOnLine.Add(linePoint1X);
                pointOnLine.Add(pointToProjectY);
            }
            else
            {
                double a1 = linePoint2X - linePoint1X;
                double a2 = linePoint2Y - linePoint1Y;
                double a3 = linePoint1Y - linePoint2Y;
                double a4 = linePoint2X - linePoint1X;
                double b1 = pointToProjectX * (linePoint2X - linePoint1X) + pointToProjectY * (linePoint2Y - linePoint1Y);
                double b2 = linePoint1Y * (linePoint2X - linePoint1X) - linePoint1X * (linePoint2Y - linePoint1Y);

                double projX = (a4 * b1 - a2 * b2) / (a1 * a4 - a2 * a3);
                double projY = (b2 - a3 * projX) / a4;

                pointOnLine.Add(Math.Round(projX, 8));
                pointOnLine.Add(Math.Round(projY, 8));
            }

            return pointOnLine;
        }

        public bool ProjectionWithinLine(double linePoint1X, double linePoint1Y, double linePoint2X, double linePoint2Y, double pointToProjectX, double pointToProjectY)
        {
            List<double> projection = ProjectPointOnLineIdentifyChangeDir(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y, pointToProjectX, pointToProjectY);

            return PointWithinLine(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y, projection[0], projection[1]);
        }

        public bool ProjectionWithinLineWithMod(double linePoint1X, double linePoint1Y, double linePoint2X, double linePoint2Y, double pointToProjectX, double pointToProjectY, double mod)
        {
            if (mod != 0) // Handles the stapling of the earth.
            {
                linePoint1X = Mod_ChangeNum(linePoint1X, mod);
                linePoint2X = Mod_ChangeNum(linePoint2X, mod);
                pointToProjectX = Mod_ChangeNum(pointToProjectX, mod);
            }
            List<double> projection = ProjectPointOnLine(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y, pointToProjectX, pointToProjectY);

            return PointWithinLine(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y, projection[0], projection[1]);
        }

        public List<double> FindProjectionWithinLine(double linePoint1X, double linePoint1Y, double linePoint2X, double linePoint2Y, double pointToProjectX, double pointToProjectY)
        {
            if (IdentifyChangeDirection_Map(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y)) // Handles the stapling of the earth.
            {
                double mod = FindNewMod(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y);
                linePoint1X = Mod_ChangeNum(linePoint1X, mod);
                linePoint2X = Mod_ChangeNum(linePoint2X, mod);
                pointToProjectX = Mod_ChangeNum(pointToProjectX, mod);
            }

            return ProjectPointOnLine(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y, pointToProjectX, pointToProjectY);
        }

        public List<double> FindProjectionWithinLineWithMod(double linePoint1X, double linePoint1Y, double linePoint2X, double linePoint2Y, double pointToProjectX, double pointToProjectY, double mod)
        {
            if (mod != 0) // Handles the stapling of the earth.
            {
                linePoint1X = Mod_ChangeNum(linePoint1X, mod);
                linePoint2X = Mod_ChangeNum(linePoint2X, mod);
                pointToProjectX = Mod_ChangeNum(pointToProjectX, mod);
            }

            return ProjectPointOnLine(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y, pointToProjectX, pointToProjectY);
        }

        public bool PointWithinLine(double linePoint1X, double linePoint1Y, double linePoint2X, double linePoint2Y, double pointToCheckLon, double pointToCheckLat)
        {
            if (IdentifyChangeDirection_Map(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y)) // Handles the stapling of the earth.
            {
                double mod = FindNewMod(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y);
                linePoint1X = Mod_ChangeNum(linePoint1X, mod);
                linePoint2X = Mod_ChangeNum(linePoint2X, mod);
                pointToCheckLon = Mod_ChangeNum(pointToCheckLon, mod);
            }
            double minLon = Math.Min(linePoint1X, linePoint2X);
            double maxLon = Math.Max(linePoint1X, linePoint2X);
            double minLat = Math.Min(linePoint1Y, linePoint2Y);
            double maxLat = Math.Max(linePoint1Y, linePoint2Y);
            if ((minLon <= pointToCheckLon) && (pointToCheckLon <= maxLon) && (minLat <= pointToCheckLat) && (pointToCheckLat <= maxLat))
                return true;

            return false;
        }

        public bool PointWithinLineWithMod(double linePoint1X, double linePoint1Y, double linePoint2X, double linePoint2Y, double pointToCheckLon, double pointToCheckLat, double mod)
        {
            if (mod != 0) // Handles the stapling of the earth.
            {
                linePoint1X = Mod_ChangeNum(linePoint1X, mod);
                linePoint2X = Mod_ChangeNum(linePoint2X, mod);
                pointToCheckLon = Mod_ChangeNum(pointToCheckLon, mod);
            }
            double minLon = Math.Min(linePoint1X, linePoint2X);
            double maxLon = Math.Max(linePoint1X, linePoint2X);
            double minLat = Math.Min(linePoint1Y, linePoint2Y);
            double maxLat = Math.Max(linePoint1Y, linePoint2Y);
            if ((minLon <= pointToCheckLon) && (pointToCheckLon <= maxLon) && (minLat <= pointToCheckLat) && (pointToCheckLat <= maxLat))
                return true;

            return false;
        }

        // Returns -1 if the projected point is before line, 0 if it is within line, and +1 if it is after the line
        public int CheckProjPointBeforeWithinAfterLine(double linePoint1X, double linePoint1Y, double linePoint2X, double linePoint2Y, double projPointToCheckLon, double projPointToCheckLat)
        {
            if (IdentifyChangeDirection_Map(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y)) // Handles the stapling of the earth.
            {
                double mod = FindNewMod(linePoint1X, linePoint1Y, linePoint2X, linePoint2Y);
                linePoint1X = Mod_ChangeNum(linePoint1X, mod);
                linePoint2X = Mod_ChangeNum(linePoint2X, mod);
                projPointToCheckLon = Mod_ChangeNum(projPointToCheckLon, mod);
            }
            double minLon = Math.Min(linePoint1X, linePoint2X);
            double maxLon = Math.Max(linePoint1X, linePoint2X);
            double minLat = Math.Min(linePoint1Y, linePoint2Y);
            double maxLat = Math.Max(linePoint1Y, linePoint2Y);
            if ((minLon <= projPointToCheckLon) && (projPointToCheckLon <= maxLon) && (minLat <= projPointToCheckLat) && (projPointToCheckLat <= maxLat))
                return 0;

            if (linePoint1X != linePoint2X)
            {
                if (projPointToCheckLon < minLon)
                {
                    if (minLon == linePoint1X)
                        return -1;
                    else
                        return 1;
                }
                else //if (projPointToCheckLon > maxLon)
                {
                    if (maxLon == linePoint1X)
                        return -1;
                    else
                        return 1;
                }
            }
            else // if (linePoint1Y != linePoint2Y)
            {
                if (projPointToCheckLat < minLat)
                {
                    if (minLat == linePoint1Y)
                        return -1;
                    else
                        return 1;
                }
                else //if (projPointToCheckLat > maxLat)
                {
                    if (maxLat == linePoint1Y)
                        return -1;
                    else
                        return 1;
                }
            }
        }

        // --- BASIC EARTH CALCULATIONS ---

        public bool IdentifyChangeDir_Simple(double lon1, double lat1, double lon2, double lat2)
        {
            double absDif = Math.Abs(lon1 - lon2);
            // Checks if it is closer to change directions.
            if (absDif < 360 - absDif)
                return false;
            else
                return true;
        }

        public bool IdentifyChangeDirection_Map(double lon1, double lat1, double lon2, double lat2)
        {
            bool changeDirection;
            double absDif = Math.Abs(lon1 - lon2);
            // Checks if it is closer to change directions.
            if (absDif < 360 - absDif)
                changeDirection = false;
            else
                changeDirection = true;

            if ((lon1 > 90) && (lon2 > 90) && ((lat1 > 65) || (lat2) > 65))
                changeDirection = true;

            if ((Math.Abs(lon1) > 177) && (Math.Abs(lon2) > 177))
                changeDirection = true;

            return changeDirection;
        }

        public double FindNewMod(double lon1, double lat1, double lon2, double lat2)
        {
            double newMod = 0;
            double smallX = 0, largeX = 0;
            if (lon1 < lon2)
            {
                smallX = lon1;
                largeX = lon2;
            }
            else
            {
                largeX = lon1;
                smallX = lon2;
            }
            //if ((smallX < 0) && (largeX >= 0))
            //    newMod = 0;
            //else
            newMod = (smallX + largeX) / 2;

            if ((lon1 > 90) && (lon2 > 90) && ((lat1 > 65) || (lat2 > 65)))
                newMod = -20;

            if ((Math.Abs(lon1) > 175) && (Math.Abs(lon2) > 175))
                newMod = 175;

            return newMod;
        }

        public double EstimateEarthRadius(double lat)
        {
            // Polar radius: 6356.752
            // Equator radius: 6378.137
            // Mean radius: 6371.008
            double radius = 6356.752;
            double difference = 21.385; // Equator - Polar radius
            radius = radius + difference * Math.Cos(lat * (Math.PI / 180));

            return radius;
        }

        public double NormalizeLon(double lon)
        {
            lon = (360 + lon) % 360;
            if (lon > 180)
                lon = lon - 360;
            else if (lon < -180)
                lon = lon + 360;

            return lon;
        }

        // --- BASIC DISTANCE CALCULATIONS ---
        public double CalcDistRL(double lon1, double lat1, double lon2, double lat2)
        {
            double diffLat = (lat1 + lat2) / 2;
            double R = EstimateEarthRadius(diffLat) / 1.852; // Convert to nm 
            double latRad1 = DegreeToRadian(lat2);
            double latRad2 = DegreeToRadian(lat1);
            double dLat = DegreeToRadian(lat1 - lat2);
            double dLon = DegreeToRadian(Math.Abs(lon1 - lon2));

            double dPhi = Math.Log(Math.Tan(latRad2 / 2 + Math.PI / 4) / Math.Tan(latRad1 / 2 + Math.PI / 4));
            double q = Math.Cos(latRad1);
            if (dPhi != 0) q = dLat / dPhi;  // E-W line gives dPhi=0
                                             // if dLon over 180° take shorter rhumb across 180° meridian:
            if (dLon > Math.PI) dLon = 2 * Math.PI - dLon;
            double dist = Math.Sqrt(dLat * dLat + q * q * dLon * dLon) * R;

            return dist;
        }

        public double CalcDistGC(double lon1, double lat1, double lon2, double lat2)
        {
            return CalcDistGC_ChangeDir(lon1, lat1, lon2, lat2, IdentifyChangeDir_Simple(lon1, lat1, lon2, lat2));
        }

        public double CalcDistGC_ChangeDir(double lon1, double lat1, double lon2, double lat2, bool changeDir)
        {
            //var R = 6371; // The mean radius of the earth in km
            double minLat = Math.Min(lat1, lat2);
            double maxLat = Math.Max(lat1, lat2);
            double difLat = maxLat - minLat;
            double meanLat = minLat + difLat / 2;
            double R = EstimateEarthRadius(meanLat); //Radius of the Earth in km
            if (changeDir)
            {
                lon1 = (360 + lon1) % 360;
                lon2 = (360 + lon2) % 360;
            }

            double dLat = DegreeToRadian(lat1 - lat2);
            double dLon = DegreeToRadian(lon1 - lon2);
            double p1LatRad = DegreeToRadian(lat2);
            double p2LatRad = DegreeToRadian(lat1);

            var a = Math.Sin(dLat / 2) * Math.Sin(dLat / 2) + Math.Sin(dLon / 2) * Math.Sin(dLon / 2) * Math.Cos(p1LatRad) * Math.Cos(p2LatRad);
            var c = 2 * Math.Atan2(Math.Sqrt(a), Math.Sqrt(1 - a));
            double distance = (R * c);
            distance = distance / 1.852; // Convert to NM

            return distance;
        }

        public double CalcDistInDeg(double lon1, double lat1, double lon2, double lat2)
        {
            return CalcDistInDeg_ChangeDir(lon1, lat1, lon2, lat2, IdentifyChangeDir_Simple(lon1, lat1, lon2, lat2));
        }

        public double CalcDistInDeg_ChangeDir(double lon1, double lat1, double lon2, double lat2, bool changeDir)
        {
            double dist = 0;

            if (!changeDir)
                dist = Math.Sqrt(Math.Pow((lon1 - lon2), 2) + Math.Pow((lat1 - lat2), 2));
            else
            {
                double difLon = 360 - Math.Max(lon1, lon2) + Math.Min(lon1, lon2); // 180 - Math.Max((double)lon1, (double)lon2) + (Math.Min((double)lon1, (double)lon2) - (-180));
                dist = Math.Sqrt(Math.Pow(difLon, 2) + Math.Pow((lat1 - lat2), 2));
            }

            return dist;
        }

         // --- RL CALCULATIONS ---

        public List<double> FindRLpointGivenDist(double lon1, double lat1, double lon2, double lat2, double distance)
        {
            return FindRLpointGivenDist_ChangeDir(lon1, lat1, lon2, lat2, distance, IdentifyChangeDir_Simple(lon1, lat1, lon2, lat2));
        }

        public List<double> FindRLpointGivenDist_ChangeDir(double lon1, double lat1, double lon2, double lat2, double distance, bool changeDir)
        {
            List<double> newPoint = new List<double>();
            if (!changeDir)
            {
                if ((lon1 == lon2) && (lat1 == lat2))
                {
                    newPoint.Add(lon1);
                    newPoint.Add(lat1);
                }
                else if (lon1 == lon2)
                {
                    newPoint.Add(lon1);
                    if (lat1 > lat2)
                        newPoint.Add(Math.Round(lat1 - distance, 6));
                    else
                        newPoint.Add(Math.Round(lat1 + distance, 6));
                }
                else if (lat1 == lat2)
                {
                    if (lon1 > lon2)
                        newPoint.Add(Math.Round(lon1 - distance, 6));
                    else
                        newPoint.Add(Math.Round(lon1 + distance, 6));
                    newPoint.Add(lat1);
                }
                else
                {
                    double foundLon, foundLat;
                    double slope = Convert.ToDouble((lat1 - lat2) / (lon1 - lon2));
                    if (lon1 > lon2)
                        foundLon = lon1 - distance / Math.Sqrt(1 + Math.Pow(slope, 2));
                    else
                        foundLon = lon1 + distance / Math.Sqrt(1 + Math.Pow(slope, 2));
                    newPoint.Add(Math.Round(foundLon, 6));
                    foundLat = lat1 + slope * (foundLon - lon1);
                    newPoint.Add(Math.Round(foundLat, 6));
                }
            }
            else
            {
                if (lat1 == lat2)
                {
                    double foundLon;
                    if (lon1 > lon2) // We reverse the argument since we have a change of directions (from -180 to +180 or vice versa)
                        foundLon = Math.Round(lon1 + distance, 6);
                    else
                        foundLon = Math.Round(lon1 - distance, 6);
                    if (foundLon > 180)
                        foundLon = foundLon - 360;
                    else if (foundLon <= -180)
                        foundLon = 360 + foundLon;
                    newPoint.Add(foundLon);
                    newPoint.Add(lat1);
                }
                else
                {
                    lon1 = (360 + lon1) % 360;
                    lon2 = (360 + lon2) % 360;
                    double foundLon, foundLat;
                    double slope = Convert.ToDouble((lat1 - lat2) / (lon1 - lon2));
                    if (lon1 > lon2)
                        foundLon = lon1 - distance / Math.Sqrt(1 + Math.Pow(slope, 2));
                    else
                        foundLon = lon1 + distance / Math.Sqrt(1 + Math.Pow(slope, 2));

                    foundLat = lat1 + slope * (foundLon - lon1);
                    if (foundLon > 180)
                        foundLon = foundLon - 360;
                    else if (foundLon <= -180)
                        foundLon = 360 + foundLon;
                    newPoint.Add(Math.Round(foundLon, 6));
                    newPoint.Add(Math.Round(foundLat, 6));
                }
            }

            return newPoint;
        }

        // --- GC CALCULATIONS ---

        public List<double> FindNewPointOnGC(double lon1, double lat1, double lon2, double lat2, double distanceNM)
        {
            double bearing = CalcBearingGC(lon1, lat1, lon2, lat2);
            List<double> newPoint = FindNewPointWithBearing(lon1, lat1, bearing, distanceNM);
            if (newPoint.Count == 2)
            {
                if (newPoint[0] > 180)
                    newPoint[0] = (newPoint[0] - 360);
                else if ((decimal)newPoint[0] < -180)
                    newPoint[0] = (360 + newPoint[0]);
            }

            return newPoint;
        }

        public double CalcBearingGC(double lon1, double lat1, double lon2, double lat2)
        {
            //Convert to radians
            double x1 = DegreeToRadian(lon1);
            double y1 = DegreeToRadian(lat1);
            double x2 = DegreeToRadian(lon2);
            double y2 = DegreeToRadian(lat2);

            double bearing = 0;
            double a = Math.Cos(y2) * Math.Sin(x2 - x1);
            double b = Math.Cos(y1) * Math.Sin(y2) - Math.Sin(y1) * Math.Cos(y2) * Math.Cos(x2 - x1);
            double adjust = 0;

            if ((a == 0) && (b == 0))
            {
                bearing = 0;
            }
            else if (b == 0)
            {
                if (a < 0)
                    bearing = 3 * Math.PI / 2;
                else
                    bearing = Math.PI / 2;
            }
            else if (b < 0)
            {
                adjust = Math.PI;
                bearing = RadianToDegree((Math.Atan(a / b) + adjust));
            }
            else
            {
                if (a < 0)
                    adjust = 2 * Math.PI;
                else
                    adjust = 0;

                bearing = RadianToDegree(Math.Atan(a / b) + adjust);
            }

            return bearing;
        }

        public List<double> FindNewPointWithBearing(double lon, double lat, double bearing, double distanceNM)
        {
            double distanceNMadjusted = distanceNM / 0.29155;
            List<double> newPoint = new List<double>();
            double x1deg = Convert.ToDouble(lon);
            double y1deg = Convert.ToDouble(lat);

            //Convert distance to km
            double distance = distanceNMadjusted / 1.852;
            //Radius of the Earth in km
            //double EARTH_RADIUS = 6371;
            double EARTH_RADIUS = EstimateEarthRadius(lat);

            //Convert to radians
            double x1 = DegreeToRadian(x1deg);
            double y1 = DegreeToRadian(y1deg);
            double bearingRadian = DegreeToRadian(bearing);

            // Convert arc distance to radians
            double c = distance / EARTH_RADIUS;

            double ynew = RadianToDegree(Math.Asin(Math.Sin(y1) * Math.Cos(c) + Math.Cos(y1) * Math.Sin(c) * Math.Cos(bearingRadian)));
            double xnew;

            double a = Math.Sin(c) * Math.Sin(bearingRadian);
            double b = Math.Cos(y1) * Math.Cos(c) - Math.Sin(y1) * Math.Sin(c) * Math.Cos(bearingRadian);

            if (b == 0)
                xnew = x1deg;
            else
                xnew = x1deg + RadianToDegree(Math.Atan(a / b));

            newPoint.Add(xnew);
            newPoint.Add(ynew);

            // Resolve accuracy issues
            if ((bearing == 0) || (bearing == 180))
                newPoint[0] = lon;
            else if ((bearing == 90) || (bearing == 270))
                newPoint[1] = lat;

            return newPoint;
        }

        // --- BASIC TRANSFORMATIONS --- 

        public double CalculateDichotomousDistanceInTriangle(double a, double b, double c)
        {
            return a * c / (b + c);
        }

        public List<double> FindDichPoint(double BDlength, double BClength, double Blon, double Blat, double Clon, double Clat)
        {
            // d = sqrt((x2-x1)^2 + (y2 - y1)^2) #distance
            // r = n / d #segment ratio
            double ratio = BDlength / BClength;
            // x3 = r * x2 + (1 - r) * x1 #find point that divides the segment
            // y3 = r * y2 + (1 - r) * y1 #into the ratio (1-r):r
            double Dlon = ratio * Blon + (1 - ratio) * Clon;
            double Dlat = ratio * Blat + (1 - ratio) * Clat;
            List<double> dichPoint = new List<double>();
            dichPoint.Add(Dlon);
            dichPoint.Add(Dlat);

            return dichPoint;
        }

        public List<double> FindDichPointChange(double BDlength, double BClength, double Blon, double Blat, double Clon, double Clat)
        {
            Blon = (360 + Blon) % 360;
            Clon = (360 + Clon) % 360;
            // d = sqrt((x2-x1)^2 + (y2 - y1)^2) #distance
            // r = n / d #segment ratio
            double ratio = BDlength / BClength;
            // x3 = r * x2 + (1 - r) * x1 #find point that divides the segment
            // y3 = r * y2 + (1 - r) * y1 #into the ratio (1-r):r
            double Dlon = ratio * Blon + (1 - ratio) * Clon;
            if (Dlon > 180)
                Dlon = -1 * (360 - Dlon);
            double Dlat = ratio * Blat + (1 - ratio) * Clat;
            List<double> dichPoint = new List<double>();
            dichPoint.Add(Dlon);
            dichPoint.Add(Dlat);

            return dichPoint;
        }

        public List<double> findPointWithDistanceOnDichotomy(List<double> A, List<double> B, List<double> D, double distance)
        {
            double mileToCoor = 0.016655;
            double totDistDec = distance * mileToCoor; // In "Accu" version, X is 2 times the required distance, check whether AX cuts any coastlines, and if not, we take point Y = AX/2
            double totDistDoub = Convert.ToDouble(totDistDec);
            double changeX = 0;
            double changeY = 0;
            List<double> X = new List<double>();
            if (A[0] == D[0]) // If the coastal point has the same lon as the found dichotomous point
            {
                X.Add(A[0]);
                if (A[1] > D[1])
                    X.Add(A[1] + totDistDec);
                else
                    X.Add(A[1] - totDistDec);
            }
            else if (A[1] == D[1])  // If the coastal point has the same lat as the found dichotomous point
            {
                if (A[0] > D[0])
                {
                    double tempX = A[0] + totDistDec;
                    if (tempX > 180)
                        tempX = -1 * (360 - tempX);
                    X.Add(tempX);
                }
                else
                {
                    double tempX = A[0] - totDistDec;
                    if (tempX < -180)
                        tempX = 360 + tempX;
                    X.Add(tempX);
                }
                X.Add(A[1]);
            }
            else
            {
                double a0 = Convert.ToDouble(A[0]), b0 = Convert.ToDouble(B[0]), d0 = Convert.ToDouble(D[0]);
                if (((Math.Sign(a0) != Math.Sign(b0)) && (Math.Abs(a0 - b0) > 180)) || ((Math.Sign(a0) != Math.Sign(d0)) && (Math.Abs(a0 - d0) > 180)) || ((Math.Sign(b0) != Math.Sign(d0)) && (Math.Abs(b0 - d0) > 180)))
                {
                    a0 = (360 + a0) % 360;
                    b0 = (360 + b0) % 360;
                    d0 = (360 + d0) % 360;
                }
                double a0b0 = b0 - a0;
                double a0d0 = d0 - a0;
                double a1b1 = Convert.ToDouble(B[1] - A[1]);
                double a1d1 = Convert.ToDouble(D[1] - A[1]);

                double cos = CalculateCosAngleOfVectors(a0b0, a1b1, a0d0, a1d1);
                changeX = Math.Abs(cos * totDistDoub);
                changeY = Math.Sqrt(Math.Pow(totDistDoub, 2) - Math.Pow(changeX, 2));
                double tempX, tempY;
                if (A[0] > D[0])
                {
                    tempX = A[0] + changeX;
                    if (tempX > 180)
                        tempX = -1 * (360 - tempX);
                }
                else
                {
                    tempX = A[0] - changeX;
                    if (tempX < -180)
                        tempX = 360 + tempX;
                }
                if (A[1] > D[1])
                    tempY = A[1] + changeY;
                else
                    tempY = A[1] - changeY;
                X.Add(tempX);
                X.Add(tempY);
            }

            // Check if the line AX cuts any coastline.
            // If it does not, keep point Y = AX/2, i.e., the point in the middle of line AX.
            // If it does, find the middle of the passage between A and point K where AX cuts the closest coastline.
            X[0] = Math.Round(X[0], 6, MidpointRounding.AwayFromZero);
            X[1] = Math.Round(X[1], 6, MidpointRounding.AwayFromZero);

            return X;
        }

        public double CalculateCosAngleOfVectors(double u1, double u2, double v1, double v2)
        {
            return (u1 * v1 + u2 * v2) / (Math.Sqrt(Math.Pow(u1, 2) + Math.Pow(u2, 2)) * Math.Sqrt(Math.Pow(v1, 2) + Math.Pow(v2, 2)));
        }

        public List<double> FindPointMidOfLine(double Alon, double Alat, double Blon, double Blat)
        {
            List<double> findPoint = new List<double>();

            findPoint.Add((Alon + Blon) / 2);
            findPoint.Add((Alat + Blat) / 2);

            return findPoint;
        }

        public DateTime UpdateDateTime(double lon1, double lat1, double lon2, double lat2, double speed, DateTime curDateTime)
        {
            double distNM = CalcDistGC(lon1, lat1, lon2, lat2);
            if (speed == 0)
                speed = 12;
            double hours = distNM / speed;
            return curDateTime.AddHours(hours);
        }

        public double CalculateVesselDirection(double vlon1, double vlat1, double vlon2, double vlat2)
        {
            double angle = 0;
            if (vlon1 == vlon2)
            {
                if (vlat1 <= vlat2)
                    angle = 0;
                else
                    angle = 180;
            }
            else
            {
                double absDif = Math.Abs(vlon1 - vlon2);
                // Checks if it is closer to change directions.
                if (absDif > 360 - absDif)
                {
                    vlon1 = (360 + vlon1) % 360;
                    vlon2 = (360 + vlon2) % 360;
                }
                if (vlat1 == vlat2)
                {
                    if (vlon1 <= vlon2)
                        angle = 90;
                    else
                        angle = 270;
                }
                else
                {
                    double division = ((vlat2 - vlat1) / (vlon2 - vlon1));
                    if (vlon1 < vlon2)
                        angle = 90 - RadianToDegree(Math.Atan(division));
                    else // if ((vlon1 > vlon2) && (vlat1 < vlat2))
                        angle = 270 - RadianToDegree(Math.Atan(division));
                }
            }

            return angle;
        }

        public List<double> FindIntersectionOfLines(List<double> line1point1, List<double> line1point2, List<double> line2point1, List<double> line2point2)
        {
            LineSlope line1slope = CalcSlope(line1point1[0], line1point1[1], line1point2[0], line1point2[1]);
            LineSlope line2slope = CalcSlope(line2point1[0], line2point1[1], line2point2[0], line2point2[1]);

            List<double> intersectP = CalculateIntersect(line1point1[0], line1point1[1], line1slope.Slope, line1slope.SlopeType, line2point1[0], line2point1[1], line2slope.Slope, line2slope.SlopeType);
            for (int i = 0; i < intersectP.Count; i++)
                intersectP[i] = Math.Round(intersectP[i], 6);

            return intersectP;
        }

        public List<double> CalculateIntersect(double point1X, double point1Y, double slope1, string slopeType1, double point2X, double point2Y, double slope2, string slopeType2)
        {
            List<double> intersect = new List<double>();

            if (!((slopeType1 == slopeType2) && (slope1 == slope2)))
            {
                if ((slopeType1 == "normal") && (slopeType2 == "normal"))
                {
                    intersect.Add((slope1 * point1X - point1Y - slope2 * point2X + point2Y) / (slope1 - slope2)); // Slope1 != Slope2 because of the "if" above.
                    intersect.Add(point1Y + slope1 * (intersect[0] - point1X));
                }
                else if ((slopeType1 == "normal") && (slopeType2 == "infint"))
                {
                    intersect.Add(point2X);
                    intersect.Add(point1Y + slope1 * (intersect[0] - point1X));
                }
                else //if ((slopeType1[0] == "infit") && (slopeType2[0] == "normal")): Both of them cannot be simultaneously "infint"; this is covered by the "if" above.
                {
                    intersect.Add(point1X);
                    intersect.Add(point2Y + slope2 * (intersect[0] - point2X));
                }
            }
            else
            {
                if ((slope1 == 0) && (slopeType1 == "normal"))
                {
                    if (point1Y == point2Y)
                    {
                        intersect.Add(point1X);
                        intersect.Add(point1Y);
                    }
                }
                else if ((slope1 == 0) && (slopeType1 == "infint"))
                {
                    if (point1X == point2X)
                    {
                        intersect.Add(point1X);
                        intersect.Add(point1Y);
                    }
                }
                else // if (slope1 != 0) 
                {
                    LineSlope lineslopeInterSect = CalcSlope(point1X, point1Y, point2X, point2Y);
                    if (lineslopeInterSect.Slope == slope1)
                    {
                        intersect.Add(point1X);
                        intersect.Add(point1Y);
                    }
                }
            }

            return intersect;
        }

        public double GetRelativeCourseDeviation(double curVesDir, double newVesDir)
        {
            double relativeDir = Math.Abs(newVesDir - curVesDir);
            if (relativeDir > 180)
                relativeDir = 360 - relativeDir;

            return relativeDir;
        }

        public bool IsInPolygon(List<List<double>> poly, List<double> point)
        {
            List<double> p1, p2;
            bool inside = false;

            if (poly.Count < 3)
                return inside;

            List<double> oldPoint = new List<double>();
            oldPoint.Add(poly[poly.Count - 1][0]);
            oldPoint.Add(poly[poly.Count - 1][1]);

            for (int i = 0; i < poly.Count; i++)
            {
                List<double> newPoint = new List<double>();
                newPoint.Add(poly[i][0]);
                newPoint.Add(poly[i][1]);

                if (newPoint[0] > oldPoint[0])
                {
                    p1 = oldPoint;
                    p2 = newPoint;
                }
                else
                {
                    p1 = newPoint;
                    p2 = oldPoint;
                }

                if (((newPoint[0] < point[0]) == (point[0] <= oldPoint[0])) && (point[1] - p1[1]) * (p2[0] - p1[0]) < (p2[1] - p1[1]) * (point[0] - p1[0]))
                    inside = !inside;

                oldPoint = newPoint;
            }

            return inside;
        }

        public List<double> CheckIntersectionBetweenLines(List<double> line1point1, List<double> line1point2, List<double> line2point1, List<double> line2point2)
        {
            List<double> intersection = new List<double>();
            List<double> intersect = FindIntersectionOfLines(line1point1, line1point2, line2point1, line2point2);
            if (intersect.Count > 0)
                // If there is an intersection that is within these two lines
                if ((((line1point1[0] <= intersect[0]) && (intersect[0] <= line1point2[0])) || ((line1point2[0] <= intersect[0]) && (intersect[0] <= line1point1[0]))) &&
                    (((line2point1[0] <= intersect[0]) && (intersect[0] <= line2point2[0])) || ((line2point2[0] <= intersect[0]) && (intersect[0] <= line2point1[0]))) &&
                    (((line1point1[1] <= intersect[1]) && (intersect[1] <= line1point2[1])) || ((line1point2[1] <= intersect[1]) && (intersect[1] <= line1point1[1]))) &&
                    (((line2point1[1] <= intersect[1]) && (intersect[1] <= line2point2[1])) || ((line2point2[1] <= intersect[1]) && (intersect[1] <= line2point1[1]))))
                {
                    intersection.Add(intersect[0]);
                    intersection.Add(intersect[1]);
                }

            return intersection;
        }

        public double[] FindLineFrom2Points(double lon1, double lat1, double lon2, double lat2)
        {
            double[] line = new double[3];

            if (lon1 == lon2)
            {
                line[0] = 0;
                line[1] = 1;
                line[2] = -1 * lon1; // x = -b, where -b = lon => b = -lon
            }
            else if (lat1 == lat2)
            {
                line[0] = 1;
                line[1] = 0;
                line[2] = lat1; // y = b
            }
            else
            {
                line[0] = 1;
                line[1] = (lat1 - lat2) / (lon1 - lon2);
                line[2] = lat1 - line[1] * lon1; // b = y_1 - ax_1
            }

            return line;
        }

        // Line y = ax + b
        // Perpendicular y = (-1/a)x + c
        // [0]: {0,1} - 0 if lon1 = lon2, else 1
        // [1]: coefficient (-1/a) = \lambda
        // [2]: coefficient c
        public double[] FindLinePerpendicularFromPoint(double[] line, double lon1, double lat1)
        {
            double[] perpendicular = new double[3];

            if (line[0] == 0) // if line is of the form x = -b, perpendicular is of the form y = c
            {
                perpendicular[0] = 1;
                perpendicular[1] = 0;
                perpendicular[2] = lat1; // y = c
            }
            else
            {
                if (line[1] == 0) // if line is of the form y = b, perpendicular is of the form x = -c
                {
                    perpendicular[0] = 0;
                    perpendicular[1] = 1;
                    perpendicular[2] = -1 * lon1; // x = -c => c = -x
                }
                else // if line is of the form y = ax + b, perpendicular is of the form y = (-1/a)x + c
                {
                    perpendicular[0] = 1;
                    perpendicular[1] = -1 / line[1];
                    perpendicular[2] = lat1 - perpendicular[1] * lon1; // c = y_1 - (-1/a)x_1 
                }
            }

            return perpendicular;
        }

        public double Mod360(double num)
        {
            return (360 + num) % 360;
        }

        public List<double> FindPointOnLineGivenDistance(double lon1, double lat1, double lon2, double lat2, double distance)
        {
            bool changeDirection;
            double absDif = Math.Abs(lon1 - lon2);
            // Checks if it is closer to change directions.
            if (absDif < 360 - absDif)
                changeDirection = false;
            else
                changeDirection = true;

            List<double> newPoint = new List<double>();
            if (!changeDirection)
            {
                if ((lon1 == lon2) && (lat1 == lat2))
                {
                    newPoint.Add(lon1);
                    newPoint.Add(lat1);
                }
                else if (lon1 == lon2)
                {
                    newPoint.Add(lon1);
                    if (lat1 > lat2)
                        newPoint.Add(Math.Round(lat1 - distance, 6));
                    else
                        newPoint.Add(Math.Round(lat1 + distance, 6));
                }
                else if (lat1 == lat2)
                {
                    if (lon1 > lon2)
                        newPoint.Add(Math.Round(lon1 - distance, 6));
                    else
                        newPoint.Add(Math.Round(lon1 + distance, 6));
                    newPoint.Add(lat1);
                }
                else
                {
                    double foundLon, foundLat;
                    double slope = Convert.ToDouble((lat1 - lat2) / (lon1 - lon2));
                    if (lon1 > lon2)
                        foundLon = lon1 - distance / Math.Sqrt(1 + Math.Pow(slope, 2));
                    else
                        foundLon = lon1 + distance / Math.Sqrt(1 + Math.Pow(slope, 2));
                    newPoint.Add(Math.Round(foundLon, 6));
                    foundLat = lat1 + slope * (foundLon - lon1);
                    newPoint.Add(Math.Round(foundLat, 6));
                }
            }
            else
            {
                if (lat1 == lat2)
                {
                    double foundLon;
                    if (lon1 > lon2) // We reverse the argument since we have a change of directions (from -180 to +180 or vice versa)
                        foundLon = Math.Round(lon1 + distance, 6);
                    else
                        foundLon = Math.Round(lon1 - distance, 6);
                    if (foundLon > 180)
                        foundLon = foundLon - 360;
                    else if (foundLon <= -180)
                        foundLon = 360 + foundLon;
                    newPoint.Add(foundLon);
                    newPoint.Add(lat1);
                }
                else
                {
                    lon1 = (360 + lon1) % 360;
                    lon2 = (360 + lon2) % 360;
                    double foundLon, foundLat;
                    double slope = Convert.ToDouble((lat1 - lat2) / (lon1 - lon2));
                    if (lon1 > lon2)
                        foundLon = lon1 - distance / Math.Sqrt(1 + Math.Pow(slope, 2));
                    else
                        foundLon = lon1 + distance / Math.Sqrt(1 + Math.Pow(slope, 2));

                    foundLat = lat1 + slope * (foundLon - lon1);
                    if (foundLon > 180)
                        foundLon = foundLon - 360;
                    else if (foundLon <= -180)
                        foundLon = 360 + foundLon;
                    newPoint.Add(Math.Round(foundLon, 6));
                    newPoint.Add(Math.Round(foundLat, 6));
                }
            }

            return newPoint;
        }
    }

    public class LineSlope
    {
        public double Slope { set; get; }
        public string SlopeType { set; get; }
    }
}