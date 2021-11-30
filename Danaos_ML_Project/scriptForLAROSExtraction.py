import csv
import cx_Oracle



class scriptForLAROSExtraction:


    def __init__(self, company, vessel, server, sid, password, usr, siteid):
        self.server = server
        self.sid = sid
        self.password = password
        self.usr = usr
        self.vessel = vessel
        self.company = company
        self.siteid = siteid
        self.timeout = 10000000


    def extractDataForSiteID(self, vessel, siteId, periodFrom, periodTo):



        dsn = cx_Oracle.makedsn(
            self.server,
            '1521',
            service_name = self.sid
        )
        connection = cx_Oracle.connect(
            user = self.usr,
            password = self.password,
            dsn=dsn
        )

        connection.callTimeout = self.timeout
        cursor_myserver = connection.cursor()

        cursor_myserver.execute(
            'select * from TBL_STATUS_PARAMETER_VAL where SITE_ID = '"'" + str(siteId) + "'"' '
                'and DATETIME >  TO_DATE( '"' "+periodFrom+" '"', '"'YYYY-MM-DD HH:MI'"') and '
                'DATETIME < TO_DATE('"' "+periodTo+" '"', '"'YYYY-MM-DD HH:MI'"') order by DATETIME')
        # if cursor_myserver.fetchall().__len__() > 0:

        rows = cursor_myserver.fetchall()
        fp = open('./pipeLineData/'+vessel+'_'+str(self.siteid)+'_.csv', 'w')
        myFile = csv.writer(fp)
        myFile.writerows(rows)
        fp.close()


def main():
    ext = scriptForLAROSExtraction("DANAOS", "CMA CGM TANCREDI", "10.2.5.80", "or19", "laros", "laros", )

    ext.extractDataForSiteID(ext.vessel, ext.siteid)

if __name__ == "__main__":
    main()
