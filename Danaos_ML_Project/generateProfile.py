import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,chisquare,chi2_contingency
import math
import seaborn as sns
#from pyproj import Proj, transform
import pyearth as sp
import matplotlib
from sklearn.cluster import KMeans
from scipy.spatial import distance as dis
from matplotlib.lines import Line2D
from decimal import Decimal
from scipy import stats
import random
from numpy import inf
#from coordinates.converter import CoordinateConverter, WGS84, L_Est97
import pyodbc
import csv
import locale
#locale.setlocale(locale.LC_NUMERIC, "en_DK.UTF-8")
import datetime
import cx_Oracle
from dateutil.rrule import rrule, DAILY, MINUTELY
from sympy.solvers import solve
from sympy import Symbol
from pathlib import Path
import itertools
from sympy import cos, sin , tan , exp , sqrt , E
from openpyxl import load_workbook
import glob, os
from pathlib import Path
from openpyxl.styles.colors import YELLOW
from openpyxl.styles import Font
from openpyxl.styles.borders import Border, Side
import shutil
from openpyxl.styles import PatternFill
from openpyxl.styles import Alignment
import matplotlib.pyplot as plt
#import seaborn as sns
from openpyxl.drawing.image import Image


class BaseProfileGenerator:

    def ConvertMSToBeaufort(self, ws):
        wsB = 0
        if ws >= 0 and ws < 0.2:
            wsB = 0
        elif ws >= 0.3 and ws < 1.5:
            wsB = 1
        elif ws >= 1.6 and ws < 3.3:
            wsB = 2
        elif ws >= 3.4 and ws < 5.4:
            wsB = 3
        elif ws >= 5.5 and ws < 7.9:
            wsB = 4

        elif ws >= 8 and ws < 10.7:
            wsB = 5
        elif ws >= 10.8 and ws < 13.8:
            wsB = 6
        elif ws >= 13.9 and ws < 17.1:
            wsB = 7
        elif ws >= 17.2 and ws < 20.7:
            wsB = 8
        elif ws >= 20.8 and ws < 24.4:
            wsB = 9
        elif ws >= 24.5 and ws < 28.4:
            wsB = 10
        elif ws >= 28.5 and ws < 32.6:
            wsB = 11
        elif ws >= 32.7:
            wsB = 12
        return wsB

    def calculateExcelStatistics(self, workbook, dtNew, velocities, draft, trim, velocitiesTlg, rawData, company,
                                 vessel, tlgDataset, period):

        if period == 'all':
            sheet = 4
        elif period == 'bdd':
            sheet = 5
        else:
            sheet = 6
        thin_border = Border(left=Side(style='thin'),
                             right=Side(style='thin'),
                             top=Side(style='thin'),
                             bottom=Side(style='thin'))

        ####STATISTICS TAB
        ##SPEED CELLS
        velocitiesTlg = np.array([k for k in tlgDataset if float(k[12]) < 50])[:, 12]
        velocitiesTlg = np.nan_to_num(velocitiesTlg.astype(float))
        velocitiesTlg = velocitiesTlg[velocitiesTlg > 0]
        velocitiesTlgAmount = velocitiesTlg.__len__()
        rowsAmount = 56
        maxSpeed = np.ceil(np.max(velocitiesTlg))
        minSpeed = np.floor(np.min(velocitiesTlg))
        i = minSpeed
        speedsApp = []
        k = 0
        workbook._sheets[sheet].row_dimensions[1].height = 35
        idh = 1

        workbook._sheets[sheet]['C' + str(idh + 2)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['C' + str(idh + 2)] = np.round(np.min(velocitiesTlg), 2)
        workbook._sheets[sheet]['C' + str(idh + 2)].font = Font(bold=True, name='Calibri', size='10.5')

        workbook._sheets[sheet]['E' + str(idh + 2)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['E' + str(idh + 2)] = np.round(np.max(velocitiesTlg), 2)
        workbook._sheets[sheet]['E' + str(idh + 2)].font = Font(bold=True, name='Calibri', size='10.5')

        while i < maxSpeed:
            workbook._sheets[sheet]['A' + str(k + 3)] = ' (' + str(i) + '-' + str(i + 0.5) + ')'
            workbook._sheets[sheet]['A' + str(k + 3)].font = Font(bold=True, name='Calibri', size='10.5')

            speedsApp.append(str(np.round(
                (np.array([k for k in velocitiesTlg if k >= i and k <= i + 0.5]).__len__()) / velocitiesTlgAmount * 100,
                2)) + '%')
            workbook._sheets[sheet]['A' + str(k + 3)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['A' + str(k + 3)].border = thin_border

            i += 0.5
            k += 1

        speedDeletedRows = 0
        if k < rowsAmount:
            # for i in range(k-1,19+1):
            workbook._sheets[sheet].delete_rows(idx=3 + k, amount=rowsAmount - k + 1)
            speedDeletedRows = rowsAmount - k + 1

        elif k > rowsAmount:
            workbook._sheets[sheet].insert_rows(idx=3 + k + 6, amount=abs(rowsAmount - k + 1))
            speedDeletedRows = -(abs(rowsAmount - k + 1))

        for i in range(0, len(speedsApp)):
            workbook._sheets[sheet]['B' + str(i + 3)] = speedsApp[i]
            workbook._sheets[sheet]['B' + str(i + 3)].alignment = Alignment(horizontal='left')

        for i in range(3, 3 + len(speedsApp)):
            try:
                if float(str(workbook._sheets[4]['B' + str(i)].value).split("%")[0]) > 2:
                    workbook._sheets[sheet]['B' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                    workbook._sheets[sheet]['A' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
            except:
                c = 0

        ##### END OF SPEED CELLS

        ##DRAFT CELLS
        draft = np.array([k for k in tlgDataset if float(k[8]) > 0 and float(k[8]) < 20])[:, 8].astype(float)
        draftAmount = draft.__len__()
        minDraft = int(np.floor(np.min(draft)))
        maxDraft = int(np.ceil(np.max(draft)))
        draftsApp = []
        k = 0
        i = minDraft
        rowsAmount = 31
        id = 70 - speedDeletedRows
        workbook._sheets[sheet].row_dimensions[id - 2].height = 35

        idh = id - 2

        workbook._sheets[sheet].merge_cells('C' + str(idh) + ':' + 'F' + str(idh))
        workbook._sheets[sheet]['C' + str(idh)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['C' + str(idh)] = 'Min and Max draft discovered'
        workbook._sheets[sheet]['C' + str(idh)].font = Font(bold=True, name='Calibri', size='10.5')
        workbook._sheets[sheet]['C' + str(idh)].border = thin_border

        workbook._sheets[sheet].merge_cells('C' + str(idh + 1) + ':' + 'D' + str(idh + 1))
        workbook._sheets[sheet]['C' + str(idh + 1)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['C' + str(idh + 1)] = 'Min'
        workbook._sheets[sheet]['C' + str(idh + 1)].font = Font(bold=True, name='Calibri', size='10.5')
        workbook._sheets[sheet]['C' + str(idh + 1)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
        workbook._sheets[sheet]['C' + str(idh + 1)].border = thin_border

        workbook._sheets[sheet].merge_cells('E' + str(idh + 1) + ':' + 'F' + str(idh + 1))
        workbook._sheets[sheet]['E' + str(idh + 1)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['E' + str(idh + 1)] = 'Max'
        workbook._sheets[sheet]['E' + str(idh + 1)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
        workbook._sheets[sheet]['E' + str(idh + 1)].font = Font(bold=True, name='Calibri', size='10.5')
        workbook._sheets[sheet]['E' + str(idh + 1)].border = thin_border

        workbook._sheets[sheet].merge_cells('C' + str(idh + 2) + ':' + 'D' + str(idh + 2))
        workbook._sheets[sheet]['C' + str(idh + 2)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['C' + str(idh + 2)] = np.round(np.min(draft), 2)
        workbook._sheets[sheet]['C' + str(idh + 2)].font = Font(bold=True, name='Calibri', size='10.5')
        workbook._sheets[sheet]['C' + str(idh + 2)].border = thin_border

        workbook._sheets[sheet].merge_cells('E' + str(idh + 2) + ':' + 'F' + str(idh + 2))
        workbook._sheets[sheet]['E' + str(idh + 2)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['E' + str(idh + 2)] = np.round(np.max(draft), 2)
        workbook._sheets[sheet]['E' + str(idh + 2)].font = Font(bold=True, name='Calibri', size='10.5')
        workbook._sheets[sheet]['E' + str(idh + 2)].border = thin_border

        while i < maxDraft:
            # workbook._sheets[4].insert_rows(k+41)
            workbook._sheets[sheet]['A' + str(k + id)] = ' (' + str(i) + '-' + str(i + 0.5) + ')'
            workbook._sheets[sheet]['A' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            draftsApp.append(str(np.round((np.array(
                [k for k in tlgDataset if float(k[8]) >= i and float(k[8]) <= i + 0.5]).__len__()) / draftAmount * 100,
                                          2)) + '%')
            workbook._sheets[sheet]['A' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['A' + str(k + id)].border = thin_border
            i += 0.5
            k += 1

        draftDeletedRows = 0
        if k < rowsAmount:
            # for i in range(k-1,19+1):
            workbook._sheets[sheet].delete_rows(idx=id + k, amount=rowsAmount - k + 1)
            draftDeletedRows = speedDeletedRows + rowsAmount - k + 1
        elif k > rowsAmount:
            workbook._sheets[sheet].insert_rows(idx=id + k, amount=rowsAmount - k + 1)
            draftDeletedRows = -(abs(speedDeletedRows + rowsAmount - k + 1))

        for i in range(0, len(draftsApp)):
            workbook._sheets[sheet]['B' + str(i + id)] = draftsApp[i]
            workbook._sheets[sheet]['B' + str(i + id)].alignment = Alignment(horizontal='left')
            height = workbook._sheets[sheet].row_dimensions[i + id].height
            if height != None:
                if float(height) > 13.8:
                    workbook._sheets[sheet].row_dimensions[i + id].height = 13.8

        for i in range(id, id + len(draftsApp)):
            try:
                if float(str(workbook._sheets[4]['B' + str(i)].value).split('%')[0]) > 2:
                    workbook._sheets[sheet]['B' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                    workbook._sheets[sheet]['A' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
            except:
                c = 0
        ##### END OF DRAFT CELSS

        ##TRIM CELLS ######################################################################
        trim = np.array([k for k in tlgDataset if float(k[17]) < 20])[:, 17].astype(float)
        trimAmount = trim.__len__()
        minTrim = int(np.floor(np.min(trim)))
        maxTrim = int(np.ceil(np.max(trim)))
        trimsApp = []
        rowsAmount = 40
        k = 0
        i = minTrim
        id = 115 - draftDeletedRows
        workbook._sheets[sheet].row_dimensions[id - 2].height = 35
        idh = id - 2

        workbook._sheets[sheet].merge_cells('C' + str(idh) + ':' + 'F' + str(idh))
        workbook._sheets[sheet]['C' + str(idh)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['C' + str(idh)] = 'Min and Max trim discovered'
        workbook._sheets[sheet]['C' + str(idh)].font = Font(bold=True, name='Calibri', size='10.5')
        workbook._sheets[sheet]['C' + str(idh)].border = thin_border

        workbook._sheets[sheet].merge_cells('C' + str(idh + 1) + ':' + 'D' + str(idh + 1))
        workbook._sheets[sheet]['C' + str(idh + 1)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['C' + str(idh + 1)] = 'Min'
        workbook._sheets[sheet]['C' + str(idh + 1)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
        workbook._sheets[sheet]['C' + str(idh + 1)].font = Font(bold=True, name='Calibri', size='10.5')
        workbook._sheets[sheet]['C' + str(idh + 1)].border = thin_border

        workbook._sheets[sheet].merge_cells('E' + str(idh + 1) + ':' + 'F' + str(idh + 1))
        workbook._sheets[sheet]['E' + str(idh + 1)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['E' + str(idh + 1)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
        workbook._sheets[sheet]['E' + str(idh + 1)] = 'Max'
        workbook._sheets[sheet]['E' + str(idh + 1)].font = Font(bold=True, name='Calibri', size='10.5')
        workbook._sheets[sheet]['E' + str(idh + 1)].border = thin_border

        workbook._sheets[sheet].merge_cells('C' + str(idh + 2) + ':' + 'D' + str(idh + 2))
        workbook._sheets[sheet]['C' + str(idh + 2)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['C' + str(idh + 2)] = np.round(np.min(trim), 2)
        workbook._sheets[sheet]['C' + str(idh + 2)].font = Font(bold=True, name='Calibri', size='10.5')
        workbook._sheets[sheet]['C' + str(idh + 2)].border = thin_border

        workbook._sheets[sheet].merge_cells('E' + str(idh + 2) + ':' + 'F' + str(idh + 2))
        workbook._sheets[sheet]['E' + str(idh + 2)].alignment = Alignment(horizontal='center')
        workbook._sheets[sheet]['E' + str(idh + 2)] = np.round(np.max(trim), 2)
        workbook._sheets[sheet]['E' + str(idh + 2)].font = Font(bold=True, name='Calibri', size='10.5')
        workbook._sheets[sheet]['E' + str(idh + 2)].border = thin_border

        while i < maxTrim:
            # workbook._sheets[sheet].insert_rows(k+27)
            if i < 0:
                workbook._sheets[sheet]['A' + str(k + id)] = ' (' + str(i) + ' - ' + str(i + 0.5) + ')'
                workbook._sheets[sheet]['A' + str(k + id)].font = Font(color='ce181e', bold=True, name='Calibri',
                                                                       size='10.5')
            else:
                workbook._sheets[sheet]['A' + str(k + id)] = ' (' + str(i) + '-' + str(i + 0.5) + ')'
                workbook._sheets[sheet]['A' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['A' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['A' + str(k + id)].border = thin_border
            trimsApp.append(str(np.round((np.array(
                [k for k in tlgDataset if float(k[16]) >= i and float(k[16]) <= i + 0.5]).__len__() / trimAmount) * 100,
                                         2)) + '%')
            i += 0.5
            k += 1

        trimDeletedRows = 0
        if k < rowsAmount:
            # for i in range(k-1,19+1):
            workbook._sheets[sheet].delete_rows(idx=id + k, amount=rowsAmount - k + 1)
            trimDeletedRows = draftDeletedRows + rowsAmount - k + 1
        if k > rowsAmount:
            workbook._sheets[sheet].insert_rows(idx=id + k, amount=rowsAmount - k + 1)
            trimDeletedRows = -(abs(draftDeletedRows + rowsAmount - k + 1))

            # speedDeletedRows+

        for i in range(0, len(trimsApp)):
            workbook._sheets[sheet]['B' + str(i + id)] = trimsApp[i]
            workbook._sheets[sheet]['B' + str(i + id)].alignment = Alignment(horizontal='left')
            height = workbook._sheets[sheet].row_dimensions[i + id].height
            if height != None:
                if float(height) > 13.8:
                    workbook._sheets[sheet].row_dimensions[i + id].height = 13.8

        for i in range(id, id + len(trimsApp)):
            try:
                if float(str(workbook._sheets[4]['B' + str(i)].value).split('%')[0]) > 2:
                    workbook._sheets[sheet]['B' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                    workbook._sheets[sheet]['A' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
            except:
                c = 0
        ##### END OF TRIM CELLS

        ##FOC TLG CELLS ######################################################################
        # foc = np.array([k for k in dtNew if float(k[16])>0])[:,16]
        foc = np.array([k for k in tlgDataset if float(k[15]) > 0 and float(k[15]) < 60 and float(k[12] > 7)])[:,
              15].astype(float)

        rowsAmount = 92
        focAmount = foc.__len__()
        minFOC = int(np.floor(np.min(foc)))
        maxFOC = int(np.ceil(np.max(foc)))
        focsApp = []
        meanSpeeds = []
        stdSpeeds = []
        ranges = []
        k = 0
        i = minFOC if minFOC > 0 else 1
        id = 171 - trimDeletedRows
        workbook._sheets[sheet].row_dimensions[id - 2].height = 35
        focsPLot = []
        speedsPlot = []
        workbook._sheets[sheet].merge_cells('C' + str(id - 2) + ':' + 'D' + str(id - 2))
        workbook._sheets[sheet]['C' + str(id - 2)].alignment = Alignment(horizontal='center')
        foc = np.array([k for k in tlgDataset if float(k[15]) > 0 and float(k[15]) < 50 and float(k[12] > 7)])
        while i < maxFOC:
            # workbook._sheets[sheet].insert_rows(k+27)
            workbook._sheets[sheet]['A' + str(k + id)] = ' (' + str(i) + '-' + str(i + 0.5) + ')'
            workbook._sheets[sheet]['A' + str(k + id)].alignment = Alignment(horizontal='center')

            workbook._sheets[sheet]['A' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['A' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['A' + str(k + id)].border = thin_border

            focArray = np.array([k for k in foc if float(k[15]) >= i and float(k[15]) <= i + 0.5])
            focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')
            meanSpeeds.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))),
                                       2) if focArray.__len__() > 0 else 'N/A')
            stdSpeeds.append(np.round((np.std(np.nan_to_num(focArray[:, 12].astype(float)))),
                                      2) if focArray.__len__() > 0 else 'N/A')

            workbook._sheets[sheet]['C' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['C' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['C' + str(k + id)].border = thin_border

            workbook._sheets[sheet]['D' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['D' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['D' + str(k + id)].border = thin_border

            if focArray.__len__() > 0:
                focsPLot.append(focArray.__len__())
                speedsPlot.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))), 2))
                ranges.append(i)
            i += 0.5
            k += 1

        xi = np.array(speedsPlot)
        yi = np.array(ranges)
        zi = np.array(focsPLot)

        plt.clf()
        # Change color with c and alpha
        p2 = np.poly1d(np.polyfit(xi, yi, 2))
        xp = np.linspace(min(xi), max(xi), 100)
        plt.plot([], [], '.', xp, p2(xp))

        plt.scatter(xi, yi, s=zi, c="red", alpha=0.4, linewidth=4)
        plt.xticks(np.arange(np.floor(min(xi)) - 1, np.ceil(max(xi)) + 1, 1))
        plt.yticks(np.arange(np.floor(min(yi)), np.ceil(max(yi)) + 1, 5))
        plt.xlabel("Speed (knots)")
        plt.ylabel("FOC (MT / day)")
        plt.title("Density plot", loc="center")
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(17.5, 9.5)

        dataModel = KMeans(n_clusters=3)
        zi = zi.reshape(-1, 1)
        dataModel.fit(zi)
        # Extract centroid values
        centroids = dataModel.cluster_centers_
        ziSorted = np.sort(centroids, axis=0)

        for z in ziSorted:
            plt.scatter([], [], c='r', alpha=0.5, s=np.floor(z[0]),
                        label=str(int(np.floor(z[0]))) + ' obs.')
        plt.legend(scatterpoints=1, frameon=True, labelspacing=2, title='# of obs')

        fig.savefig('./Figures/' + company + '_' + vessel + str(sheet) + '_1.png', dpi=96)
        # plt.clf()

        img = Image('./Figures/' + company + '_' + vessel + str(sheet) + '_1.png')

        workbook._sheets[sheet].add_image(img, 'F' + str(id - 2))
        # workbook._sheets[4].insert_image('G'+str(id+30), './Danaos_ML_Project/Figures/'+company+'_'+vessel+'.png', {'x_scale': 0.5, 'y_scale': 0.5})

        if k < rowsAmount:
            # for i in range(k-1,19+1):
            workbook._sheets[sheet].delete_rows(idx=id + k, amount=rowsAmount - k + 1)
            focDeletedRows = rowsAmount - k + 1

        for i in range(0, len(focsApp)):
            workbook._sheets[sheet]['B' + str(i + id)] = focsApp[i]

            workbook._sheets[sheet]['B' + str(i + id)].alignment = Alignment(horizontal='center')

            workbook._sheets[sheet]['C' + str(i + id)] = meanSpeeds[i]
            workbook._sheets[sheet]['D' + str(i + id)] = stdSpeeds[i]

            workbook._sheets[sheet]['C' + str(i + id)].alignment = Alignment(horizontal='center')
            workbook._sheets[sheet]['D' + str(i + id)].alignment = Alignment(horizontal='center')

            workbook._sheets[sheet]['B' + str(i + id)].border = thin_border
            workbook._sheets[sheet]['B' + str(i + id)].font = Font(name='Calibri', size='10.5')

            height = workbook._sheets[sheet].row_dimensions[i + id].height
            if height != None:
                if float(height) > 13.8:
                    workbook._sheets[sheet].row_dimensions[i + id].height = 13.8

        for i in range(id, id + len(focsApp)):
            try:
                if float(str(workbook._sheets[4]['B' + str(i)].value).split('%')[0]) > 1.5:
                    workbook._sheets[sheet]['B' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                    workbook._sheets[sheet]['A' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                    workbook._sheets[sheet]['C' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                    workbook._sheets[sheet]['D' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
            except:
                c = 0
        ##### END OF FOC TLG CELLS
        ########################################################################
        ##FOC PER MILE / SPEED RANGES / BEAUFORTS GRAPHS
        focPerMile = np.array([k for k in tlgDataset if float(k[17]) > 0])[:, 17].astype(float)
        steamTimeSum = np.sum(np.array([k for k in tlgDataset if float(k[17]) > 0])[:, 13].astype(float) + np.array(
            [k for k in tlgDataset if float(k[17]) > 0])[:, 14].astype(float) / 60)
        # minsSlc = np.sum()
        # steamTimeSum = hoursSlc + minsSlc/60
        rowsAmount = 92
        focAmount = focPerMile.__len__()
        minFOC = int(np.floor(np.min(focPerMile)))
        maxFOC = int(np.ceil(np.max(focPerMile)))
        focsApp = []
        meanSpeeds = []
        steamTimeRange = []
        minSlc = []
        hoursSlc = []
        stdSpeeds = []
        ranges = []
        k = 0
        i = minFOC
        id = 415
        workbook._sheets[sheet].row_dimensions[id - 2].height = 35
        focsPLot = []
        speedsPlot = []
        workbook._sheets[sheet].merge_cells('C' + str(id - 2) + ':' + 'D' + str(id - 2))
        workbook._sheets[sheet]['C' + str(id - 2)].alignment = Alignment(horizontal='center')
        while i < maxFOC:
            # workbook._sheets[sheet].insert_rows(k+27)
            i = np.round(i, 2)
            workbook._sheets[4]['A' + str(k + id)] = ' (' + str(i) + '-' + str(i + 0.02) + ')'
            workbook._sheets[sheet]['A' + str(k + id)].alignment = Alignment(horizontal='center')

            workbook._sheets[sheet]['A' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['A' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['A' + str(k + id)].border = thin_border
            focArray = np.array([k for k in tlgDataset if float(k[17]) >= i and float(k[17]) <= i + 0.02])
            if focArray.__len__() > 0:
                hoursSlc.append(focArray[:, 13])
                minSlc.append(focArray[:, 14])
                steamTimeRange = np.sum(np.array(focArray[:, 13].astype(float) + focArray[:, 14].astype(float) / 60))

                focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')
                meanSpeeds.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))),
                                           2) if focArray.__len__() > 0 else 'N/A')
                stdSpeeds.append(np.round((np.std(np.nan_to_num(focArray[:, 12].astype(float)))),
                                          2) if focArray.__len__() > 0 else 'N/A')

            workbook._sheets[4]['C' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['C' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['C' + str(k + id)].border = thin_border

            workbook._sheets[sheet]['D' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['D' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['D' + str(k + id)].border = thin_border

            if focArray.__len__() > 0:
                focsPLot.append(np.round(steamTimeRange / steamTimeSum, 4))
                # focsPLot.append(focArray.__len__())
                speedsPlot.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))), 2))
                ranges.append(i)
            i += 0.02
            k += 1

        plt.clf()
        xi = np.array(speedsPlot)
        yi = np.array(ranges)
        zi = np.array(focsPLot)
        # Change color with c and alpha
        p2 = np.poly1d(np.polyfit(xi, yi, 2))
        xp = np.linspace(min(xi), max(xi), 100)
        plt.plot([], [], '.', xp, p2(xp))

        plt.scatter(xi, yi, s=zi * 2000, c="red", alpha=0.4, linewidth=4)
        plt.xticks(np.arange(np.floor(min(xi)) - 1, np.ceil(max(xi)) + 1, 1))
        # plt.yticks(np.arange(np.floor(min(yi)), np.ceil(max(yi)) + 1, 5))
        plt.xlabel("Speed (knots)")
        plt.ylabel("FOC (per / Mile)")
        plt.title("Density plot", loc="center")
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(17.5, 9.5)

        dataModel = KMeans(n_clusters=3)
        zi = zi.reshape(-1, 1)
        dataModel.fit(zi)
        # Extract centroid values
        centroids = dataModel.cluster_centers_
        ziSorted = np.sort(centroids, axis=0)

        for z in ziSorted:
            plt.scatter([], [], c='r', alpha=0.5, s=z[0] * 700,
                        label=str(np.round(z[0], 3)) + ' Hours')
        plt.legend(scatterpoints=1, frameon=True, labelspacing=2, title=' Hours')

        fig.savefig('./Figures/' + company + '_' + vessel + '_3.png', dpi=96)

        img = Image('./Figures/' + company + '_' + vessel + '_3.png')

        workbook._sheets[4].add_image(img, 'F' + str(415))
        x = 0

        for i in range(0, len(focsApp)):
            workbook._sheets[sheet]['B' + str(i + id)] = focsApp[i]

            workbook._sheets[sheet]['B' + str(i + id)].alignment = Alignment(horizontal='center')

            workbook._sheets[sheet]['C' + str(i + id)] = meanSpeeds[i]
            workbook._sheets[sheet]['D' + str(i + id)] = stdSpeeds[i]

            workbook._sheets[sheet]['C' + str(i + id)].alignment = Alignment(horizontal='center')
            workbook._sheets[sheet]['D' + str(i + id)].alignment = Alignment(horizontal='center')

            workbook._sheets[sheet]['B' + str(i + id)].border = thin_border
            workbook._sheets[sheet]['B' + str(i + id)].font = Font(name='Calibri', size='10.5')

            height = workbook._sheets[sheet].row_dimensions[i + id].height
            if height != None:
                if float(height) > 13.8:
                    workbook._sheets[sheet].row_dimensions[i + id].height = 13.8

        for i in range(id, id + len(focsApp)):
            try:
                if float(str(workbook._sheets[4]['B' + str(i)].value).split('%')[0]) > 1.5:
                    workbook._sheets[sheet]['B' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                    workbook._sheets[sheet]['A' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                    workbook._sheets[sheet]['C' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                    workbook._sheets[sheet]['D' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
            except:
                o = 0
        ##########################FOC PER WIND RANGES / WIND ANGLE
        foc = np.array([k for k in tlgDataset if float(k[15]) > 0 and float(k[15]) < 40])[:, 15].astype(float)

        focAmount = foc.__len__()
        minFOC = int(np.floor(np.min(foc)))
        maxFOC = int(np.ceil(np.max(foc)))

        focsAppGen = []
        meanSpeedsGen = []
        stdSpeedsGen = []
        rangesGen = []
        focsPLotGen = []
        speedsPlotGen = []

        focsApp = []
        meanSpeeds = []
        stdSpeeds = []
        ranges = []
        focsPLot = []
        speedsPlot = []

        focsApp1 = []
        meanSpeeds1 = []
        stdSpeeds1 = []
        ranges1 = []
        focsPLot1 = []
        speedsPlot1 = []

        focsApp2 = []
        meanSpeeds2 = []
        stdSpeeds2 = []
        ranges2 = []
        focsPLot2 = []
        speedsPlot2 = []

        focsApp3 = []
        meanSpeeds3 = []
        stdSpeeds3 = []
        ranges3 = []
        focsPLot3 = []
        speedsPlot3 = []

        focsApp4 = []
        meanSpeeds4 = []
        stdSpeeds4 = []
        ranges4 = []
        focsPLot4 = []
        speedsPlot4 = []

        k = 0
        i = minFOC if minFOC > 0 else 1
        id = 171 - trimDeletedRows
        # workbook._sheets[sheet].row_dimensions[id - 2].height = 35

        # workbook._sheets[sheet].merge_cells('C' + str(id - 2) + ':' + 'D' + str(id - 2))
        # workbook._sheets[sheet]['C' + str(id - 2)].alignment = Alignment(horizontal='center')
        while i < maxFOC:
            # workbook._sheets[sheet].insert_rows(k+27)
            '''workbook._sheets[4]['A' + str(k + id)] = ' (' + str(i) + '-' + str(i + 0.5) + ')'
            workbook._sheets[sheet]['A' + str(k + id)].alignment = Alignment(horizontal='center')

            workbook._sheets[sheet]['A' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['A' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['A' + str(k + id)].border = thin_border'''
            focArrayGen = np.array([k for k in tlgDataset if
                                    float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 1 and float(
                                        k[11]) <= 3])

            focArray = np.array([k for k in focArrayGen if
                                 float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 1 and float(
                                     k[11]) <= 3 and
                                 float(k[10]) >= 0 and float(k[10]) <= 22.5])

            focArray1 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 1 and float(
                                      k[11]) <= 3 and
                                  float(k[10]) >= 22.5 and float(k[10]) <= 67.5])

            focArray2 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 1 and float(
                                      k[11]) <= 3 and
                                  float(k[10]) >= 67.5 and float(k[10]) <= 112.5])

            focArray3 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 1 and float(
                                      k[11]) <= 3 and float(k[10]) >= 112.5 and float(k[10]) <= 157.5])

            focArray4 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 1 and float(
                                      k[11]) <= 3 and float(k[10]) >= 157.5 and float(k[10]) <= 180])

            focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')
            meanSpeeds.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))),
                                       2) if focArray.__len__() > 0 else 'N/A')
            stdSpeeds.append(np.round((np.std(np.nan_to_num(focArray[:, 12].astype(float)))),
                                      2) if focArray.__len__() > 0 else 'N/A')

            '''workbook._sheets[4]['C' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['C' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['C' + str(k + id)].border = thin_border

            workbook._sheets[sheet]['D' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['D' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['D' + str(k + id)].border = thin_border'''
            if focArrayGen.__len__() > 0:
                focsPLotGen.append(focArrayGen.__len__())
                speedsPlotGen.append(np.round((np.mean(np.nan_to_num(focArrayGen[:, 12].astype(float)))), 2))
                rangesGen.append(np.round((np.mean(np.nan_to_num(focArrayGen[:, 15].astype(float)))), 2))

            if focArray.__len__() > 0:
                focsPLot.append(focArray.__len__())
                speedsPlot.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))), 2))
                ranges.append(np.round((np.mean(np.nan_to_num(focArray[:, 15].astype(float)))), 2))

            if focArray1.__len__() > 0:
                focsPLot1.append(focArray1.__len__())
                speedsPlot1.append(np.round((np.mean(np.nan_to_num(focArray1[:, 12].astype(float)))), 2))
                ranges1.append(np.round((np.mean(np.nan_to_num(focArray1[:, 15].astype(float)))), 2))

            if focArray2.__len__() > 0:
                focsPLot2.append(focArray2.__len__())
                speedsPlot2.append(np.round((np.mean(np.nan_to_num(focArray2[:, 12].astype(float)))), 2))
                ranges2.append(np.round((np.mean(np.nan_to_num(focArray2[:, 15].astype(float)))), 2))

            if focArray3.__len__() > 0:
                focsPLot3.append(focArray3.__len__())
                speedsPlot3.append(np.round((np.mean(np.nan_to_num(focArray3[:, 12].astype(float)))), 2))
                ranges3.append(np.round((np.mean(np.nan_to_num(focArray3[:, 15].astype(float)))), 2))

            if focArray4.__len__() > 0:
                focsPLot4.append(focArray4.__len__())
                speedsPlot4.append(np.round((np.mean(np.nan_to_num(focArray4[:, 12].astype(float)))), 2))
                ranges4.append(np.round((np.mean(np.nan_to_num(focArray4[:, 15].astype(float)))), 2))

            i += 0.5
            k += 1

        plt.clf()
        xiGen = np.array(speedsPlotGen)
        yiGen = np.array(rangesGen)
        ziGen = np.array(focsPLotGen)
        # Change color with c and alpha
        p2Gen = np.poly1d(np.polyfit(xiGen, yiGen, 2))
        xpGen = np.linspace(min(xiGen), max(xiGen), 100)
        # plt.plot([], [], '.', xpGen, p2Gen(xpGen), color='black')

        # plt.scatter(xiGen, yiGen, s=ziGen , c="red", alpha=0.4, linewidth=4)
        #########################################################
        xi = np.array(speedsPlot)
        yi = np.array(ranges)
        zi = np.array(focsPLot)
        # Change color with c and alpha
        # p2 = np.poly1d(np.polyfit(xi, yi, 2))
        # xp = np.linspace(min(xi), max(xi), 100)
        # line1=plt.plot([], [], '.', xp, p2(xp),color='red')

        plt.scatter(xi, yi, s=zi * 10, c="red", alpha=0.4, linewidth=4)
        # plt.xticks(np.arange(np.floor(min(xi)) - 1, np.ceil(max(xi)) + 1, 1))
        # plt.yticks(np.arange(np.floor(min(yi)), np.ceil(max(yi)) + 1, 5))
        #####################################################################
        xi1 = np.array(speedsPlot1)
        yi1 = np.array(ranges1)
        zi1 = np.array(focsPLot1)
        # Change color with c and alpha
        # p21 = np.poly1d(np.polyfit(xi1, yi1, 2))
        # xp1 = np.linspace(min(xi1), max(xi1), 100)
        # line2=plt.plot([], [], '.', xp1, p21(xp1),color='green')

        plt.scatter(xi1, yi1, s=zi1 * 10, c="green", alpha=0.4, linewidth=4)
        # plt.xticks(np.arange(np.floor(min(xi1)) - 1, np.ceil(max(xi1)) + 1, 1))
        # plt.yticks(np.arange(np.floor(min(yi1)), np.ceil(max(yi1)) + 1, 5))
        ###################################################################################################################
        #####################################################################
        xi2 = np.array(speedsPlot2)
        yi2 = np.array(ranges2)
        zi2 = np.array(focsPLot2)
        # Change color with c and alpha
        # p22 = np.poly1d(np.polyfit(xi2, yi2, 2))
        # xp2 = np.linspace(min(xi2), max(xi2), 100)
        # plt.plot([], [], '.', xp2, p22(xp2),color='blue')
        plt.scatter(xi2, yi2, s=zi2 * 10, c="blue", alpha=0.4, linewidth=4)
        ###################################################################################################################
        xi3 = np.array(speedsPlot3)
        yi3 = np.array(ranges3)
        zi3 = np.array(focsPLot3)
        # Change color with c and alpha
        # p23 = np.poly1d(np.polyfit(xi3, yi3, 2))
        # xp3 = np.linspace(min(xi3), max(xi3), 100)
        # plt.plot([], [], '.', xp3, p23(xp3), color='orange')
        plt.scatter(xi3, yi3, s=zi3 * 10, c="orange", alpha=0.4, linewidth=4)
        ###################################################################################################################
        if speedsPlot4.__len__() > 0:
            xi4 = np.array(speedsPlot4)
            yi4 = np.array(ranges4)
            zi4 = np.array(focsPLot4)
            # Change color with c and alpha
            p24 = np.poly1d(np.polyfit(xi4, yi4, 2))
            xp4 = np.linspace(min(xi4), max(xi4), 100)
            # plt.plot([], [], '.', xp4, p24(xp4), color='purple')
            plt.scatter(xi4, yi4, s=zi4 * 10, c="purple", alpha=0.4, linewidth=4)
            # plt.xticks(np.arange(np.floor(min(xi2)) - 1, np.ceil(max(xi2)) + 1, 1))
            plt.yticks(np.arange(minFOC, maxFOC + 1, 5))
        ###################################################################################################################

        ############################################
        ############################################
        ############################################
        plt.xlabel("Speed (knots)")
        plt.ylabel("FOC (MT / day) ")
        plt.title("Density plot for Wind Speed (1 - 3) Beauforts", loc="center")
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(17.5, 9.5)

        dataModel = KMeans(n_clusters=3)
        ziGen = ziGen.reshape(-1, 1)
        dataModel.fit(ziGen)
        # Extract centroid values
        centroids = dataModel.cluster_centers_
        ziSorted = np.sort(centroids, axis=0)

        for z in ziSorted:
            plt.scatter([], [], c='grey', alpha=0.5, s=np.floor(z[0] * 10),
                        label=str(int(np.floor(z[0]))) + ' obs.')
        legend1 = plt.legend(scatterpoints=1, frameon=True, labelspacing=2, title='# of obs', loc='upper right')

        custom_lines = [Line2D([0], [0], color='red', lw=4),
                        Line2D([0], [0], color='green', lw=4),
                        Line2D([0], [0], color='blue', lw=4),
                        Line2D([0], [0], color='orange', lw=4),
                        Line2D([0], [0], color='purple', lw=4)]

        plt.legend(custom_lines, ['(0 - 22.5)', '(22.5 - 67.5)', '(67.5 - 112.5)', '(112.5 - 157.5)', '(157.5 - 180)'],
                   title='Wind Dir', loc='upper left')

        plt.gca().add_artist(legend1)

        fig.savefig('./Figures/' + company + '_' + vessel + str(sheet) + '_4.png', dpi=96)

        # plt.clf()

        img = Image('./Figures/' + company + '_' + vessel + str(sheet) + '_4.png')
        workbook._sheets[sheet].add_image(img, 'A' + str(193))
        x = 0
        ###BEAFUORTS (3-5) #############################################################
        focsAppGen = []
        meanSpeedsGen = []
        stdSpeedsGen = []
        rangesGen = []
        focsPLotGen = []
        speedsPlotGen = []

        focsApp = []
        meanSpeeds = []
        stdSpeeds = []
        ranges = []
        focsPLot = []
        speedsPlot = []

        focsApp1 = []
        meanSpeeds1 = []
        stdSpeeds1 = []
        ranges1 = []
        focsPLot1 = []
        speedsPlot1 = []

        focsApp2 = []
        meanSpeeds2 = []
        stdSpeeds2 = []
        ranges2 = []
        focsPLot2 = []
        speedsPlot2 = []

        focsApp3 = []
        meanSpeeds3 = []
        stdSpeeds3 = []
        ranges3 = []
        focsPLot3 = []
        speedsPlot3 = []

        focsApp4 = []
        meanSpeeds4 = []
        stdSpeeds4 = []
        ranges4 = []
        focsPLot4 = []
        speedsPlot4 = []

        k = 0
        i = minFOC if minFOC > 0 else 1
        id = 171 - trimDeletedRows
        # workbook._sheets[sheet].row_dimensions[id - 2].height = 35

        # workbook._sheets[sheet].merge_cells('C' + str(id - 2) + ':' + 'D' + str(id - 2))
        # workbook._sheets[sheet]['C' + str(id - 2)].alignment = Alignment(horizontal='center')
        while i < maxFOC:
            # workbook._sheets[sheet].insert_rows(k+27)
            '''workbook._sheets[4]['A' + str(k + id)] = ' (' + str(i) + '-' + str(i + 0.5) + ')'
            workbook._sheets[sheet]['A' + str(k + id)].alignment = Alignment(horizontal='center')

            workbook._sheets[sheet]['A' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['A' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['A' + str(k + id)].border = thin_border'''
            focArrayGen = np.array([k for k in tlgDataset if
                                    float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 3 and float(
                                        k[11]) <= 5])

            focArray = np.array([k for k in focArrayGen if
                                 float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 3 and float(
                                     k[11]) <= 5 and
                                 float(k[10]) >= 0 and float(k[10]) <= 22.5])

            focArray1 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 3 and float(
                                      k[11]) <= 5 and
                                  float(k[10]) >= 22.5 and float(k[10]) <= 67.5])

            focArray2 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 3 and float(
                                      k[11]) <= 5 and
                                  float(k[10]) >= 67.5 and float(k[10]) <= 112.5])

            focArray3 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 3 and float(
                                      k[11]) <= 5 and float(k[10]) >= 112.5 and float(k[10]) <= 157.5])

            focArray4 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 3 and float(
                                      k[11]) <= 5 and float(k[10]) >= 157.5 and float(k[10]) <= 180])

            focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')
            meanSpeeds.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))),
                                       2) if focArray.__len__() > 0 else 'N/A')
            stdSpeeds.append(np.round((np.std(np.nan_to_num(focArray[:, 12].astype(float)))),
                                      2) if focArray.__len__() > 0 else 'N/A')

            '''workbook._sheets[4]['C' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['C' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['C' + str(k + id)].border = thin_border

            workbook._sheets[sheet]['D' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['D' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['D' + str(k + id)].border = thin_border'''
            if focArrayGen.__len__() > 0:
                focsPLotGen.append(focArrayGen.__len__())
                speedsPlotGen.append(np.round((np.mean(np.nan_to_num(focArrayGen[:, 12].astype(float)))), 2))
                rangesGen.append(np.round((np.mean(np.nan_to_num(focArrayGen[:, 15].astype(float)))), 2))

            if focArray.__len__() > 0:
                focsPLot.append(focArray.__len__())
                speedsPlot.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))), 2))
                ranges.append(np.round((np.mean(np.nan_to_num(focArray[:, 15].astype(float)))), 2))

            if focArray1.__len__() > 0:
                focsPLot1.append(focArray1.__len__())
                speedsPlot1.append(np.round((np.mean(np.nan_to_num(focArray1[:, 12].astype(float)))), 2))
                ranges1.append(np.round((np.mean(np.nan_to_num(focArray1[:, 15].astype(float)))), 2))

            if focArray2.__len__() > 0:
                focsPLot2.append(focArray2.__len__())
                speedsPlot2.append(np.round((np.mean(np.nan_to_num(focArray2[:, 12].astype(float)))), 2))
                ranges2.append(np.round((np.mean(np.nan_to_num(focArray2[:, 15].astype(float)))), 2))

            if focArray3.__len__() > 0:
                focsPLot3.append(focArray3.__len__())
                speedsPlot3.append(np.round((np.mean(np.nan_to_num(focArray3[:, 12].astype(float)))), 2))
                ranges3.append(np.round((np.mean(np.nan_to_num(focArray3[:, 15].astype(float)))), 2))

            if focArray4.__len__() > 0:
                focsPLot4.append(focArray4.__len__())
                speedsPlot4.append(np.round((np.mean(np.nan_to_num(focArray4[:, 12].astype(float)))), 2))
                ranges4.append(np.round((np.mean(np.nan_to_num(focArray4[:, 15].astype(float)))), 2))

            i += 0.5
            k += 1

        plt.clf()
        xiGen = np.array(speedsPlotGen)
        yiGen = np.array(rangesGen)
        ziGen = np.array(focsPLotGen)
        # Change color with c and alpha
        p2Gen = np.poly1d(np.polyfit(xiGen, yiGen, 2))
        xpGen = np.linspace(min(xiGen), max(xiGen), 100)
        # plt.plot([], [], '.', xpGen, p2Gen(xpGen), color='black')

        # plt.scatter(xiGen, yiGen, s=ziGen , c="red", alpha=0.4, linewidth=4)
        #########################################################
        xi = np.array(speedsPlot)
        yi = np.array(ranges)
        zi = np.array(focsPLot)
        # Change color with c and alpha
        p2 = np.poly1d(np.polyfit(xi, yi, 2))
        xp = np.linspace(min(xi), max(xi), 100)
        # line1=plt.plot([], [], '.', xp, p2(xp),color='red')

        plt.scatter(xi, yi, s=zi * 10, c="red", alpha=0.4, linewidth=4)
        # plt.xticks(np.arange(np.floor(min(xi)) - 1, np.ceil(max(xi)) + 1, 1))
        # plt.yticks(np.arange(np.floor(min(yi)), np.ceil(max(yi)) + 1, 5))
        #####################################################################
        xi1 = np.array(speedsPlot1)
        yi1 = np.array(ranges1)
        zi1 = np.array(focsPLot1)
        # Change color with c and alpha
        p21 = np.poly1d(np.polyfit(xi1, yi1, 2))
        xp1 = np.linspace(min(xi1), max(xi1), 100)
        # line2=plt.plot([], [], '.', xp1, p21(xp1),color='green')

        plt.scatter(xi1, yi1, s=zi1 * 10, c="green", alpha=0.4, linewidth=4)
        # plt.xticks(np.arange(np.floor(min(xi1)) - 1, np.ceil(max(xi1)) + 1, 1))
        # plt.yticks(np.arange(np.floor(min(yi1)), np.ceil(max(yi1)) + 1, 5))
        ###################################################################################################################
        #####################################################################
        xi2 = np.array(speedsPlot2)
        yi2 = np.array(ranges2)
        zi2 = np.array(focsPLot2)
        # Change color with c and alpha
        p22 = np.poly1d(np.polyfit(xi2, yi2, 2))
        xp2 = np.linspace(min(xi2), max(xi2), 100)
        # plt.plot([], [], '.', xp2, p22(xp2),color='blue')
        plt.scatter(xi2, yi2, s=zi2 * 10, c="blue", alpha=0.4, linewidth=4)
        ###################################################################################################################
        xi3 = np.array(speedsPlot3)
        yi3 = np.array(ranges3)
        zi3 = np.array(focsPLot3)
        # Change color with c and alpha
        p23 = np.poly1d(np.polyfit(xi3, yi3, 2))
        xp3 = np.linspace(min(xi3), max(xi3), 100)
        # plt.plot([], [], '.', xp3, p23(xp3), color='orange')
        plt.scatter(xi3, yi3, s=zi3 * 10, c="orange", alpha=0.4, linewidth=4)
        ###################################################################################################################
        if speedsPlot4.__len__() > 0:
            xi4 = np.array(speedsPlot4)
            yi4 = np.array(ranges4)
            zi4 = np.array(focsPLot4)
            # Change color with c and alpha
            p24 = np.poly1d(np.polyfit(xi4, yi4, 2))
            xp4 = np.linspace(min(xi4), max(xi4), 100)
            # plt.plot([], [], '.', xp4, p24(xp4), color='purple')
            plt.scatter(xi4, yi4, s=zi4 * 10, c="purple", alpha=0.4, linewidth=4)
            # plt.xticks(np.arange(np.floor(min(xi2)) - 1, np.ceil(max(xi2)) + 1, 1))
            plt.yticks(np.arange(minFOC, maxFOC + 1, 5))
        ###################################################################################################################

        ############################################
        ############################################
        ############################################
        plt.xlabel("Speed (knots)")
        plt.ylabel("FOC (MT / day) ")
        plt.title("Density plot for Wind Speed (3 - 5) Beauforts", loc="center")
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(17.5, 9.5)

        dataModel = KMeans(n_clusters=3)
        ziGen = ziGen.reshape(-1, 1)
        dataModel.fit(ziGen)
        # Extract centroid values
        centroids = dataModel.cluster_centers_
        ziSorted = np.sort(centroids, axis=0)

        for z in ziSorted:
            plt.scatter([], [], c='grey', alpha=0.5, s=np.floor(z[0] * 10),
                        label=str(int(np.floor(z[0]))) + ' obs.')
        legend1 = plt.legend(scatterpoints=1, frameon=True, labelspacing=2, title='# of obs', loc='upper right')

        custom_lines = [Line2D([0], [0], color='red', lw=4),
                        Line2D([0], [0], color='green', lw=4),
                        Line2D([0], [0], color='blue', lw=4),
                        Line2D([0], [0], color='orange', lw=4),
                        Line2D([0], [0], color='purple', lw=4)]

        plt.legend(custom_lines, ['(0 - 22.5)', '(22.5 - 67.5)', '(67.5 - 112.5)', '(112.5 - 157.5)', '(157.5 - 180)'],
                   title='Wind Dir', loc='upper left')

        plt.gca().add_artist(legend1)

        fig.savefig('./Figures/' + company + '_' + vessel + str(sheet) + '_5.png', dpi=96)

        # plt.clf()

        img = Image('./Figures/' + company + '_' + vessel + str(sheet) + '_5.png')
        workbook._sheets[sheet].add_image(img, 'T' + str(193))
        ##################### BEAUFORTS (5-8)
        ###BEAFUORTS (3-5) #############################################################
        focsAppGen = []
        meanSpeedsGen = []
        stdSpeedsGen = []
        rangesGen = []
        focsPLotGen = []
        speedsPlotGen = []

        focsApp = []
        meanSpeeds = []
        stdSpeeds = []
        ranges = []
        focsPLot = []
        speedsPlot = []

        focsApp1 = []
        meanSpeeds1 = []
        stdSpeeds1 = []
        ranges1 = []
        focsPLot1 = []
        speedsPlot1 = []

        focsApp2 = []
        meanSpeeds2 = []
        stdSpeeds2 = []
        ranges2 = []
        focsPLot2 = []
        speedsPlot2 = []

        focsApp3 = []
        meanSpeeds3 = []
        stdSpeeds3 = []
        ranges3 = []
        focsPLot3 = []
        speedsPlot3 = []

        focsApp4 = []
        meanSpeeds4 = []
        stdSpeeds4 = []
        ranges4 = []
        focsPLot4 = []
        speedsPlot4 = []

        k = 0
        i = minFOC if minFOC > 0 else 1
        id = 171 - trimDeletedRows
        # workbook._sheets[sheet].row_dimensions[id - 2].height = 35

        # workbook._sheets[sheet].merge_cells('C' + str(id - 2) + ':' + 'D' + str(id - 2))
        # workbook._sheets[sheet]['C' + str(id - 2)].alignment = Alignment(horizontal='center')
        while i < maxFOC:
            # workbook._sheets[sheet].insert_rows(k+27)
            '''workbook._sheets[4]['A' + str(k + id)] = ' (' + str(i) + '-' + str(i + 0.5) + ')'
            workbook._sheets[sheet]['A' + str(k + id)].alignment = Alignment(horizontal='center')

            workbook._sheets[sheet]['A' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['A' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['A' + str(k + id)].border = thin_border'''
            focArrayGen = np.array([k for k in tlgDataset if
                                    float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 5 and float(
                                        k[11]) <= 9])

            focArray = np.array([k for k in focArrayGen if
                                 float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 5 and float(
                                     k[11]) <= 9 and
                                 float(k[10]) >= 0 and float(k[10]) <= 22.5])

            focArray1 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 5 and float(
                                      k[11]) <= 9 and
                                  float(k[10]) >= 22.5 and float(k[10]) <= 67.5])

            focArray2 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 5 and float(
                                      k[11]) <= 9 and
                                  float(k[10]) >= 67.5 and float(k[10]) <= 112.5])

            focArray3 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 5 and float(
                                      k[11]) <= 9 and float(k[10]) >= 112.5 and float(k[10]) <= 157.5])

            focArray4 = np.array([k for k in focArrayGen if
                                  float(k[15]) >= i and float(k[15]) <= i + 0.5 and float(k[11]) >= 5 and float(
                                      k[11]) <= 9 and float(k[10]) >= 157.5 and float(k[10]) <= 180])

            focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')
            meanSpeeds.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))),
                                       2) if focArray.__len__() > 0 else 'N/A')
            stdSpeeds.append(np.round((np.std(np.nan_to_num(focArray[:, 12].astype(float)))),
                                      2) if focArray.__len__() > 0 else 'N/A')

            '''workbook._sheets[4]['C' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['C' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['C' + str(k + id)].border = thin_border

            workbook._sheets[sheet]['D' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
            workbook._sheets[sheet]['D' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
            workbook._sheets[sheet]['D' + str(k + id)].border = thin_border'''
            if focArrayGen.__len__() > 0:
                focsPLotGen.append(focArrayGen.__len__())
                speedsPlotGen.append(np.round((np.mean(np.nan_to_num(focArrayGen[:, 12].astype(float)))), 2))
                rangesGen.append(np.round((np.mean(np.nan_to_num(focArrayGen[:, 15].astype(float)))), 2))

            if focArray.__len__() > 0:
                focsPLot.append(focArray.__len__())
                speedsPlot.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))), 2))
                ranges.append(np.round((np.mean(np.nan_to_num(focArray[:, 15].astype(float)))), 2))

            if focArray1.__len__() > 0:
                focsPLot1.append(focArray1.__len__())
                speedsPlot1.append(np.round((np.mean(np.nan_to_num(focArray1[:, 12].astype(float)))), 2))
                ranges1.append(np.round((np.mean(np.nan_to_num(focArray1[:, 15].astype(float)))), 2))

            if focArray2.__len__() > 0:
                focsPLot2.append(focArray2.__len__())
                speedsPlot2.append(np.round((np.mean(np.nan_to_num(focArray2[:, 12].astype(float)))), 2))
                ranges2.append(np.round((np.mean(np.nan_to_num(focArray2[:, 15].astype(float)))), 2))

            if focArray3.__len__() > 0:
                focsPLot3.append(focArray3.__len__())
                speedsPlot3.append(np.round((np.mean(np.nan_to_num(focArray3[:, 12].astype(float)))), 2))
                ranges3.append(np.round((np.mean(np.nan_to_num(focArray3[:, 15].astype(float)))), 2))

            if focArray4.__len__() > 0:
                focsPLot4.append(focArray4.__len__())
                speedsPlot4.append(np.round((np.mean(np.nan_to_num(focArray4[:, 12].astype(float)))), 2))
                ranges4.append(np.round((np.mean(np.nan_to_num(focArray4[:, 15].astype(float)))), 2))

            i += 0.5
            k += 1

        plt.clf()
        xiGen = np.array(speedsPlotGen)
        yiGen = np.array(rangesGen)
        ziGen = np.array(focsPLotGen)
        # Change color with c and alpha
        p2Gen = np.poly1d(np.polyfit(xiGen, yiGen, 2))
        xpGen = np.linspace(min(xiGen), max(xiGen), 100)
        # plt.plot([], [], '.', xpGen, p2Gen(xpGen), color='black')

        # plt.scatter(xiGen, yiGen, s=ziGen , c="red", alpha=0.4, linewidth=4)
        #########################################################
        xi = np.array(speedsPlot)
        yi = np.array(ranges)
        zi = np.array(focsPLot)
        # Change color with c and alpha
        p2 = np.poly1d(np.polyfit(xi, yi, 2))
        xp = np.linspace(min(xi), max(xi), 100)
        # line1=plt.plot([], [], '.', xp, p2(xp),color='red')

        plt.scatter(xi, yi, s=zi * 10, c="red", alpha=0.4, linewidth=4)
        # plt.xticks(np.arange(np.floor(min(xi)) - 1, np.ceil(max(xi)) + 1, 1))
        # plt.yticks(np.arange(np.floor(min(yi)), np.ceil(max(yi)) + 1, 5))
        #####################################################################
        xi1 = np.array(speedsPlot1)
        yi1 = np.array(ranges1)
        zi1 = np.array(focsPLot1)
        # Change color with c and alpha
        p21 = np.poly1d(np.polyfit(xi1, yi1, 2))
        xp1 = np.linspace(min(xi1), max(xi1), 100)
        # line2=plt.plot([], [], '.', xp1, p21(xp1),color='green')

        plt.scatter(xi1, yi1, s=zi1 * 10, c="green", alpha=0.4, linewidth=4)
        # plt.xticks(np.arange(np.floor(min(xi1)) - 1, np.ceil(max(xi1)) + 1, 1))
        # plt.yticks(np.arange(np.floor(min(yi1)), np.ceil(max(yi1)) + 1, 5))
        ###################################################################################################################
        #####################################################################
        xi2 = np.array(speedsPlot2)
        yi2 = np.array(ranges2)
        zi2 = np.array(focsPLot2)
        # Change color with c and alpha
        p22 = np.poly1d(np.polyfit(xi2, yi2, 2))
        xp2 = np.linspace(min(xi2), max(xi2), 100)
        # plt.plot([], [], '.', xp2, p22(xp2),color='blue')
        plt.scatter(xi2, yi2, s=zi2 * 10, c="blue", alpha=0.4, linewidth=4)
        ###################################################################################################################
        if speedsPlot3.__len__() > 0:
            xi3 = np.array(speedsPlot3)
            yi3 = np.array(ranges3)
            zi3 = np.array(focsPLot3)
            # Change color with c and alpha
            p23 = np.poly1d(np.polyfit(xi3, yi3, 2))
            xp3 = np.linspace(min(xi3), max(xi3), 100)
            # plt.plot([], [], '.', xp3, p23(xp3), color='orange')
            plt.scatter(xi3, yi3, s=zi3 * 10, c="orange", alpha=0.4, linewidth=4)
        ###################################################################################################################
        if speedsPlot4.__len__() > 0:
            xi4 = np.array(speedsPlot4)
            yi4 = np.array(ranges4)
            zi4 = np.array(focsPLot4)
            # Change color with c and alpha
            p24 = np.poly1d(np.polyfit(xi4, yi4, 2))
            xp4 = np.linspace(min(xi4), max(xi4), 100)
            # plt.plot([], [], '.', xp4, p24(xp4), color='purple')
            plt.scatter(xi4, yi4, s=zi4 * 10, c="purple", alpha=0.4, linewidth=4)
            # plt.xticks(np.arange(np.floor(min(xi2)) - 1, np.ceil(max(xi2)) + 1, 1))
            plt.yticks(np.arange(minFOC, maxFOC + 1, 5))
        ###################################################################################################################

        ############################################
        ############################################
        ############################################
        plt.xlabel("Speed (knots)")
        plt.ylabel("FOC (MT / day) ")
        plt.title("Density plot for Wind Speed (5 - 8) Beauforts", loc="center")
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(17.5, 9.5)

        dataModel = KMeans(n_clusters=3)
        ziGen = ziGen.reshape(-1, 1)
        dataModel.fit(ziGen)
        # Extract centroid values
        centroids = dataModel.cluster_centers_
        ziSorted = np.sort(centroids, axis=0)

        for z in ziSorted:
            plt.scatter([], [], c='grey', alpha=0.5, s=np.floor(z[0] * 10),
                        label=str(int(np.floor(z[0]))) + ' obs.')
        legend1 = plt.legend(scatterpoints=1, frameon=True, labelspacing=2, title='# of obs', loc='upper right')

        custom_lines = [Line2D([0], [0], color='red', lw=4),
                        Line2D([0], [0], color='green', lw=4),
                        Line2D([0], [0], color='blue', lw=4),
                        Line2D([0], [0], color='orange', lw=4),
                        Line2D([0], [0], color='purple', lw=4)]

        plt.legend(custom_lines, ['(0 - 22.5)', '(22.5 - 67.5)', '(67.5 - 112.5)', '(112.5 - 157.5)', '(157.5 - 180)'],
                   title='Wind Dir', loc='upper left')

        plt.gca().add_artist(legend1)

        fig.savefig('./Figures/' + company + '_' + vessel + str(sheet) + '_6.png', dpi=96)

        plt.clf()

        img = Image('./Figures/' + company + '_' + vessel + str(sheet) + '_6.png')
        workbook._sheets[sheet].add_image(img, 'H' + str(248))
        if rawData:
            ##############START OF RAW TAB ####################################################################
            ##############START OF RAW TAB ####################################################################
            ###START OF STW RAW #####################
            ##SPEED  RAW CELLS
            rowsAmount = 56
            velocities = np.array([k for k in dtNew])[:, 12]
            velocities = np.nan_to_num(velocities.astype(float))
            velocities = velocities[velocities >= 0]
            velocitiesAmount = velocities.__len__()
            maxSpeed = np.ceil(np.max(velocities))
            minSpeed = np.floor(np.min(velocities))
            speedsApp = []
            id = 3
            k = 0
            i = minSpeed
            idh = 1

            workbook._sheets[5]['C' + str(idh + 2)].alignment = Alignment(horizontal='center')
            workbook._sheets[5]['C' + str(idh + 2)] = np.round(np.min(velocities), 2)
            workbook._sheets[5]['C' + str(idh + 2)].font = Font(bold=True, name='Calibri', size='10.5')

            workbook._sheets[5]['E' + str(idh + 2)].alignment = Alignment(horizontal='center')
            workbook._sheets[5]['E' + str(idh + 2)] = np.round(np.max(velocities), 2)
            workbook._sheets[5]['E' + str(idh + 2)].font = Font(bold=True, name='Calibri', size='10.5')

            workbook._sheets[5].row_dimensions[1].height = 35
            focPerMile = []
            while i < maxSpeed:
                workbook._sheets[5]['A' + str(k + 3)] = ' (' + str(i) + '-' + str(i + 0.5) + ')'
                workbook._sheets[5]['A' + str(k + 3)].font = Font(bold=True, name='Calibri', size='10.5')

                meanFoc = np.array([k for k in dtNew if float(k[12]) >= i and float(k[12]) <= i + 0.5])
                if meanFoc.__len__() > 0:
                    meanSpeed = i + 0.5
                    focPerMile.append(np.round(np.mean(meanFoc[:, 15]) / (meanSpeed), 2))
                speedsApp.append(str(np.round((np.array([k for k in dtNew if float(k[12]) >= i and float(
                    k[12]) <= i + 0.5]).__len__()) / velocitiesAmount * 100, 2)) + '%')
                i += 0.5
                k += 1

            speedDeletedRows = 0
            if k < rowsAmount:
                # for i in range(k-1,19+1):
                workbook._sheets[5].delete_rows(idx=3 + k, amount=rowsAmount - k + 1)
                speedDeletedRows = rowsAmount - k + 1
            for i in range(0, len(focPerMile)):
                workbook._sheets[5]['G' + str(i + 4)] = focPerMile[i]

            for i in range(0, len(speedsApp)):
                workbook._sheets[5]['B' + str(i + 3)] = speedsApp[i]
                workbook._sheets[5]['B' + str(k + 3)].alignment = Alignment(horizontal='left')
                height = workbook._sheets[5].row_dimensions[i + 3].height
                if height != None:
                    if float(height) > 13.8:
                        workbook._sheets[5].row_dimensions[i + 3].height = 13.8

            for i in range(3, 3 + len(speedsApp)):
                try:
                    if float(workbook._sheets[5]['B' + str(i)].value) > 500:
                        workbook._sheets[5]['B' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                        workbook._sheets[5]['A' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                except:
                    c = 0

            for i in range(id, id + len(speedsApp)):
                if float(str(workbook._sheets[5]['B' + str(i)].value).split('%')[0]) > 1.5:
                    workbook._sheets[5]['B' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                    workbook._sheets[5]['A' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")

            ##### END OF SPEED RAW CELLS

            ##FOC RAW CELLS ######################################################################
            foc = np.array([k for k in dtNew if float(k[15]) > 0 and float(k[15] < 45)])[:, 15]
            rowsAmount = 92
            focAmount = foc.__len__()
            minFOC = int(np.floor(np.min(foc)))
            maxFOC = int(np.ceil(np.max(foc)))
            focsApp = []
            k = 0
            i = minFOC
            id = 171 - speedDeletedRows
            focsPLot = []
            speedsPlot = []
            focPerMile = []
            ranges = []
            meanSpeeds = []
            stdSpeeds = []
            workbook._sheets[5].merge_cells('C' + str(id - 2) + ':' + 'D' + str(id - 2))
            workbook._sheets[5]['C' + str(id - 2)].alignment = Alignment(horizontal='center')
            workbook._sheets[5].row_dimensions[id - 2].height = 35

            while i < maxFOC:
                workbook._sheets[5]['A' + str(k + id)] = ' (' + str(i) + '-' + str(i + 0.5) + ')'
                workbook._sheets[5]['A' + str(k + id)].alignment = Alignment(horizontal='center')

                workbook._sheets[5]['A' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
                workbook._sheets[5]['A' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
                workbook._sheets[5]['A' + str(k + id)].border = thin_border
                focArray = np.array([k for k in dtNew if float(k[15]) >= i and float(k[15]) <= i + 0.5])
                focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')
                meanSpeeds.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))),
                                           2) if focArray.__len__() > 0 else 'N/A')
                stdSpeeds.append(np.round((np.std(np.nan_to_num(focArray[:, 12].astype(float)))),
                                          2) if focArray.__len__() > 0 else 'N/A')
                meanFoc = np.mean(focArray[:, 15])
                meanSpeed = np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))), 2)
                focPerMile.append(np.round(meanFoc / (meanSpeed * 24), 2))
                workbook._sheets[5]['C' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
                workbook._sheets[5]['C' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
                workbook._sheets[5]['C' + str(k + id)].border = thin_border

                workbook._sheets[5]['D' + str(k + id)].font = Font(bold=True, name='Calibri', size='10.5')
                workbook._sheets[5]['D' + str(k + id)].fill = PatternFill(fgColor='dae3f3', fill_type="solid")
                workbook._sheets[5]['D' + str(k + id)].border = thin_border

                if focArray.__len__() > 0:
                    focsPLot.append(focArray.__len__())
                    speedsPlot.append(np.round((np.mean(np.nan_to_num(focArray[:, 12].astype(float)))), 2))
                    ranges.append(i)
                i += 0.5
                k += 1

            if k < rowsAmount:
                # for i in range(k-1,19+1):
                workbook._sheets[5].delete_rows(idx=id + k, amount=rowsAmount - k + 1)
                focDeletedRows = rowsAmount - k + 1

            for i in range(0, len(focsApp)):

                workbook._sheets[5]['E' + str(i + id)] = focPerMile[i]

                workbook._sheets[5]['B' + str(i + id)] = focsApp[i]

                workbook._sheets[5]['B' + str(i + id)].alignment = Alignment(horizontal='center')

                workbook._sheets[5]['C' + str(i + id)] = meanSpeeds[i]
                workbook._sheets[5]['D' + str(i + id)] = stdSpeeds[i]

                workbook._sheets[5]['C' + str(i + id)].alignment = Alignment(horizontal='center')
                workbook._sheets[5]['D' + str(i + id)].alignment = Alignment(horizontal='center')

                workbook._sheets[5]['B' + str(i + id)].border = thin_border
                workbook._sheets[5]['B' + str(i + id)].font = Font(name='Calibri', size='10.5')
                height = workbook._sheets[5].row_dimensions[i + id].height
                if height != None:
                    if float(height) > 13.8:
                        workbook._sheets[5].row_dimensions[i + id].height = 13.8

            for i in range(id, id + len(focsApp)):
                try:
                    if float(str(workbook._sheets[5]['B' + str(i)].value).split('%')[0]) > 1.5:
                        workbook._sheets[5]['B' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                        workbook._sheets[5]['A' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                        workbook._sheets[5]['C' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                        workbook._sheets[5]['D' + str(i)].fill = PatternFill(fgColor=YELLOW, fill_type="solid")
                except:
                    c = 0

            plt.clf()

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(17.5, 9.5)

            xi = np.array(speedsPlot)
            yi = np.array(ranges)
            zi = np.array(focsPLot)
            # Change color with c and alpha
            p2 = np.poly1d(np.polyfit(xi, yi, 3))
            xp = np.linspace(min(xi), max(xi), 100)
            plt.plot([], [], '.', xp, p2(xp))

            plt.scatter(xi, yi, s=zi, c="red", alpha=0.4, linewidth=4)
            plt.xticks(np.arange(np.floor(min(xi)), np.ceil(max(xi)) + 1, 1))
            plt.yticks(np.arange(min(yi), max(yi) + 1, 5))
            plt.xlabel("Speed (knots)")
            plt.ylabel("FOC (MT / day)")
            plt.title("Density plot", loc="center")

            dataModel = KMeans(n_clusters=3)
            zi = zi.reshape(-1, 1)
            dataModel.fit(zi)
            # Extract centroid values
            centroids = dataModel.cluster_centers_
            ziSorted = np.sort(centroids, axis=0)

            for z in ziSorted:
                plt.scatter([], [], c='r', alpha=0.5, s=np.floor(z[0]),
                            label='       ' + str(int(np.floor(z[0]))) + ' obs.')
            plt.legend(borderpad=4, scatterpoints=1, frameon=True, labelspacing=6, title='# of obs')

            fig.savefig('./Figures/' + company + '_' + vessel + '_2.png', dpi=96)
            # plt.clf()
            img = Image('./Figures/' + company + '_' + vessel + '_2.png')

            workbook._sheets[5].add_image(img, 'F' + str(id - 2))
            ##### END OF FOC RAW CELLS

        ########################################################## END OF STATISTICS TAB #########################################
        return workbook

    def fillExcelProfCons(self, company, vessel, pathToexcel, dataSet, rawData, tlgDataset, dataSetBDD, dataSetADD):
        ##FEATURE SET EXACT POSITION OF COLUMNS NEEDED IN ORDER TO PRODUCE EXCEL
        # 2nd place BALLAST FLAG
        # 8th place DRAFT
        # 10th place WD
        # 11th place WF
        # 12th place SPEED
        # 15th place ME FOC 24H
        # 16th place ME FOC 24H TLGS
        # 17th place TRIM
        # 19th place SteamHours
        # 18th place STW_TLG

        # if float(k[5])>6.5
        wd = np.array([k for k in dataSet])[:, 10]
        for i in range(0, len(wd)):
            if wd[i] > 180:
                wd[i] = wd[i] - 180  # and  float(k[8])<20
        dataSet[:, 10] = wd
        lenConditionTlg = 5
        dtNew = np.array([k for k in dataSet if float(k[15]) > 0 and float(k[12]) > 0])  # and  float(k[8])<20
        dtNewBDD = np.array([k for k in dataSetBDD if float(k[15]) > 0 and float(k[12]) > 0])
        dtNewADD = np.array([k for k in dataSetADD if float(k[15]) > 0 and float(k[12]) > 0])

        ballastDt = np.array([k for k in dtNew if k[2] == 'B' if float(k[8]) < 16])[:, 7:].astype(float)
        ladenDt = np.array([k for k in dtNew if k[2] == 'L' if float(k[8]) < 16])[:, 7:].astype(float)

        meanDraftBallast = round(float(np.mean(np.array([k for k in ballastDt if k[1] > 0])[:, 1])), 2)
        meanDraftLadden = round(float(np.mean(np.array([k for k in ladenDt if k[1] > 0])[:, 1])), 2)
        minDraftLadden = round(float(np.min(np.array([k for k in ladenDt if k[1] > 0])[:, 1])), 2)
        maxDraftBallast = round(float(np.max(np.array([k for k in ballastDt if k[1] > 0])[:, 1])), 2)

        for i in range(0, len(dtNew)):
            # tNew[i, 10] = self.getRelativeDirectionWeatherVessel(float(dtNew[i, 7]), float(dtNew[i, 10]))
            if str(dtNew[i, 2]) == 'nan' and float(dtNew[i, 8]) > 0:
                if float(dtNew[i, 8]) >= meanDraftLadden:
                    dtNew[i, 2] = 'L'
                else:
                    # if float(dtNew[i, 8]) <=meanDraftBallast+1:
                    dtNew[i, 2] = 'B'
        ########################################################################
        ########################################################################
        ########################################################################

        ballastDt = np.array([k for k in dtNew if k[2] == 'B' if float(k[8]) > 0])[:, 7:].astype(float)
        ladenDt = np.array([k for k in dtNew if k[2] == 'L' if float(k[8]) > 0])[:, 7:].astype(float)

        meanDraftBallast = round(float(np.mean(np.array([k for k in ballastDt if k[1] > 0])[:, 1])), 2)
        meanDraftLadden = round(float(np.mean(np.array([k for k in ladenDt if k[1] > 0])[:, 1])), 2)
        minDraftLadden = round(float(np.min(np.array([k for k in ladenDt if k[1] > 0])[:, 1])), 2)
        maxDraftBallast = round(float(np.max(np.array([k for k in ballastDt if k[1] > 0])[:, 1])), 2)

        draft = (np.array((np.array([k for k in dtNew if float(k[8]) > 0 and float(k[8]) < 20])[:, 8])).astype(float))
        trim = (np.array((np.array([k for k in dtNew if float(k[17]) < 20])[:, 17])).astype(float))
        velocities = (
            np.array((np.array([k for k in dtNew if float(k[12]) > 0])[:, 12])).astype(float))  # and float(k[12]) < 18
        velocitiesTlg = (
            np.array((np.array([k for k in dtNew if float(k[18]) > 0 and float(k[18]) < 35])[:, 12])).astype(float))
        '''if tlgDataset==[]:
              tlgDataset = dtNew
              tlgDatasetBDD = dtNewBDD
              tlgDatasetADD = dtNewADD

              #velocitiesTlgBDD = (
                  #np.array((np.array([k for k in dtNewBDD if float(k[18]) > 0 ])[:, 12])).astype(float))
              #velocitiesTlgADD = (
                  #np.array((np.array([k for k in dtNewADD if float(k[18]) > 0])[:, 12])).astype(float))
          else:
              velocitiesTlg = (np.array((np.array([k for k in dtNew if float(k[18]) > 0 and float(k[18])<35 ])[:, 18])).astype(float)) #and float(k[12]) < 18'''

        dataModel = KMeans(n_clusters=4)
        velocities = velocities.reshape(-1, 1)
        dataModel.fit(velocities)
        # Extract centroid values
        centroids = dataModel.cluster_centers_
        velocitiesSorted = np.sort(centroids, axis=0)
        ################################################################################################
        ballastDt = np.array([k for k in dtNew if k[2] == 'B'])[:, 7:].astype(float)
        ladenDt = np.array([k for k in dtNew if k[2] == 'L'])[:, 7:].astype(float)

        velocitiesB = np.array([k for k in ballastDt if k[5] > 6 and k[5] < 16])[:, 5]

        dataModel = KMeans(n_clusters=4)
        velocitiesB = velocitiesB.reshape(-1, 1)
        dataModel.fit(velocitiesB)
        labels = dataModel.predict(velocitiesB)
        # Extract centroid values

        centroidsB = dataModel.cluster_centers_
        centroidsB = np.sort(centroidsB, axis=0)
        ##LOAD EXCEL
        workbook = load_workbook(filename=pathToexcel)

        wind = [0, 22.5, 67.5, 112.5, 157.5, 180]

        ###SEA STATE
        ballastMaxSeaSate = []
        for i in range(0, len(wind) - 1):
            arrayWF = np.array(
                [k for k in ballastDt if k[3] >= wind[i] and k[3] <= wind[i + 1] and k[4] <= 9 and k[8] > 0])
            maxSS = np.max(arrayWF[:, 4]) if arrayWF.__len__() > 0 else 0
            ballastMaxSeaSate.append(round(maxSS, 2))

        for i in range(3, 8):
            workbook._sheets[3]['B' + str(i)] = ballastMaxSeaSate[i - 3]
        ####################################################
        laddenMaxSeaSate = []
        for i in range(0, len(wind) - 1):
            arrayWF = np.array(
                [k for k in ladenDt if k[3] >= wind[i] and k[3] <= wind[i + 1] and k[4] <= 9 and k[8] > 0])
            maxSS = np.max(arrayWF[:, 4]) if arrayWF.__len__() > 0 else 0
            laddenMaxSeaSate.append(round(maxSS, 2))

        for i in range(12, 17):
            workbook._sheets[3]['B' + str(i)] = laddenMaxSeaSate[i - 12]
        ###############################
        ##end of sea state
        ###
        # meanDraftBallast = round(float(np.mean(np.array([k for k in ballastDt if k[1] > 0])[:, 1])), 2)
        workbook._sheets[2]['B2'] = np.floor(meanDraftBallast) + 0.5
        # meanDraftLadden = round(float(np.mean(np.array([k for k in ladenDt if k[1] > 0])[:, 1])), 2)
        workbook._sheets[1]['B2'] = np.ceil(meanDraftLadden) + 0.5

        ##WRITE DRAFTS TO FILE
        drafts = []
        drafts.append(np.floor(meanDraftBallast) + 0.5)
        drafts.append(np.ceil(meanDraftLadden) + 0.5)
        with open('./data/' + company + '/' + vessel + '/ListOfDrafts.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(
                ['Draft'])
            for i in range(0, 2):
                data_writer.writerow(
                    [drafts[i]])

        #############################################################################
        ##BALLAST
        ##delete ballast outliers
        # np.delete(ladenDt, [i for (i, v) in enumerate(ladenDt[:, 8]) if v < (
        # np.mean(ladenDt[:, 8]) - np.std(ladenDt[:, 8])) or v > np.mean(
        # ladenDt[:, 8]) + np.std(
        # ladenDt[:, 8])], 0)
        partitionsX = []
        partitionLabels = []
        # For each label
        for curLbl in np.unique(labels):
            # Create a partition for X using records with corresponding label equal to the current
            partitionsX.append(np.asarray(velocitiesB[labels == curLbl]))
            # Create a partition for Y using records with corresponding label equal to the current

            # Keep partition label to ascertain same order of results
            partitionLabels.append(curLbl)

        sorted = []
        initialX = len(partitionsX)
        initialXpart = partitionsX
        while sorted.__len__() < initialX:
            min = 100000000000
            for i in range(0, len(partitionsX)):
                mean = np.mean(partitionsX[i])
                if mean < min:
                    min = mean
                    minIndx = i
            sorted.append(partitionsX[minIndx])
            # partitionsX.remove(partitionsX[minIndx])
            partitionsX.pop(minIndx)

        ################################################################################################################

        workbook = self.calculateExcelStatistics(workbook, dtNew, velocities, draft, trim, velocitiesTlg, rawData,
                                                 company, vessel, tlgDataset, 'all')
        # workbook = self.calculateExcelStatistics(workbook, dtNewBDD, velocities, draft, trim, velocitiesTlgBDD, rawData,
        # company, vessel, tlgDatasetBDD,'bdd')
        # workbook = self.calculateExcelStatistics(workbook, dtNewADD, velocities, draft, trim, velocitiesTlgBDD, rawData,
        # company, vessel, tlgDatasetADD, 'Add')
        ##delete ladden outliers
        # np.delete(ladenDt, [i for (i, v) in enumerate(ladenDt[:, 8]) if v < (
        # np.mean(ladenDt[:, 8]) - np.std(ladenDt[:, 8])) or v > np.mean(
        # ladenDt[:, 8]) + np.std(
        # ladenDt[:, 8])], 0)

        ###SPEED 10 WIND <1.5
        vel0Min = np.floor(np.min(sorted[0])) + 0.5
        vel1Min = np.floor(np.min(sorted[1])) + 0.5
        vel2Min = np.floor(np.min(sorted[2])) + 0.5
        vel3Min = np.floor(np.min(sorted[3])) + 0.5

        vel0Max = np.floor(np.max(sorted[0])) + 0.5
        vel1Max = np.floor(np.max(sorted[1])) + 0.5
        vel2Max = np.floor(np.max(sorted[2])) + 0.5
        vel3Max = np.floor(np.max(sorted[3])) + 0.5

        vel0Mean = np.floor(np.mean(sorted[0])) + 0.5
        vel1Mean = np.floor(np.mean(sorted[1])) + 0.5
        vel2Mean = np.floor(np.mean(sorted[2])) + 0.5
        vel3Mean = np.floor(np.mean(sorted[3])) + 0.5

        ####VESSEL BASIC INFO
        blVelocities = []
        blVelocities.append(vel0Mean)
        blVelocities.append(vel1Mean)
        blVelocities.append(vel2Mean)
        blVelocities.append(vel3Mean)

        '''vel0Min = 3
          vel1Min = 12.5
          vel2Min =12.5
          vel3Min = 13.5

          vel0Max = 12
          vel1Max = 13
          vel2Max = 13.5
          vel3Max = 18'''
        # vel0Min = 6
        # vel0Max = 9

        '''workbook._sheets[2]['B6'] = str(vel0Mean) + '  ('+str(vel0Min)+' - '+str(vel0Max)+')'
          workbook._sheets[2]['B16'] = str(vel1Mean) + '  ('+str(vel1Min)+' - '+str(vel1Max)+')'
          workbook._sheets[2]['B26'] = str(vel2Mean) + '  ('+str(vel2Min)+' - '+str(vel2Max)+')'
          workbook._sheets[2]['B36'] = str(vel3Mean) + '  ('+str(vel3Min)+' - '+str(vel3Max)+')'''

        workbook._sheets[0]['B2'] = vessel
        workbook._sheets[0]['B5'] = round(velocitiesSorted[3][0], 1)
        workbook._sheets[0]['B7'] = np.min(draft)
        workbook._sheets[0]['B8'] = np.max(draft)
        ##END OF VESSEL BASIC INFO

        ballastDt10_0 = []
        numberOfApp10_0 = []

        minAccThres = 0

        FocCentral = np.array([k for k in ballastDt if
                               k[5] >= vel0Min and k[5] <= vel0Max and k[8] > 10])[:, 8]

        steamTimeGen = np.array([k for k in ballastDt if
                                 k[5] >= vel0Min and k[5] <= vel0Max and k[8] > 10])[:, 12]

        weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

        centralMean = np.mean(np.array([k for k in ballastDt if
                                        k[5] >= vel0Min and k[5] <= vel0Max and k[8] > 10])[:, 8])
        centralMean = weighted_avgFocCentral
        centralArray = np.array([k for k in ballastDt if
                                 k[5] >= vel0Min and k[5] <= vel0Max and k[8] > 10])[:, 8]
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 0 and k[4] <= 1 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[i] and
                                 k[3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 0 and k[4] <= 1 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k >= 3])

            # tlgarrayFoc=np.mean(np.array([k for k in ballastDt if k[5] >= round(centroidsB[0][0], 2) and k[5] <= 10 and k[4] >= 0 and k[4] <= 1 and k[9] > 10])[:, 9])
            tlgarrayFoc = np.array([k for k in ballastDt if k[5] >= vel0Min and k[5] <= vel0Max and k[8] > 10])

            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array([k for k in ballastDt if k[5] >= vel0Min and k[5] <= vel0Max and k[8] > 10])[:,
                              9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp10_0.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8], weights=steamTime[:, 12
                                                                          ]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                numberOfApp10_0.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt10_0.append(round(meanFoc, 2))

        for i in range(9, 14):
            workbook._sheets[2]['B' + str(i)] = ballastDt10_0[i - 9]

        ###SPEED 10  2 < WIND <3
        ballastDt10_3 = []
        numberOfApp10_3 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 2 and k[4] <= 3 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[i] and
                                 k[
                                     3] <= wind[i + 1] and k[
                                     8] > 5])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 2 and k[4] <= 3 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 5])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 13])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])
            if tlgarrayFoc.__len__() > lenConditionTlg:

                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp10_3.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                numberOfApp10_3.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt10_3.append(round(meanFoc, 2))

        for i in range(9, 14):
            workbook._sheets[2]['C' + str(i)] = ballastDt10_3[i - 9]

        ###SPEED 10  4 < WIND <5
        ballastDt10_5 = []
        numberOfApp10_5 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 4 and k[4] <= 5 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[i] and
                                 k[
                                     3] <= wind[i + 1] and k[
                                     8] > 5])
            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 4 and k[4] <= 5 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 5])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 13])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])[:, 9]

                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp10_5.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                numberOfApp10_5.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt10_5.append(round(meanFoc, 2))

        for i in range(9, 14):
            workbook._sheets[2]['D' + str(i)] = ballastDt10_5[i - 9]

        ###SPEED 10  7 < WIND <8
        ballastDt10_8 = []
        numberOfApp10_8 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 6 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[i] and
                                 k[3] <= wind[i + 1] and k[
                                     8] > 5])
            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 6 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 5])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 13])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp10_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp10_8.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt10_8.append(round(meanFoc, 2))

        for i in range(9, 14):
            workbook._sheets[2]['E' + str(i)] = ballastDt10_8[i - 9]

        ##################################################################################################################
        ##################################################################################################################

        ###SPEED 11.5   WIND <1.5

        centralMean = np.mean(np.array([k for k in ballastDt if
                                        k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 8])
        centralArray = np.array([k for k in ballastDt if
                                 k[5] >= vel1Min and k[5] <= vel1Max and k[8] > 4])[:, 8]
        ballastDt11_0 = []
        numberOfApp11_0 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 0 and k[4] <= 1 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 5])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 0 and k[4] <= 1 and k[5] >= vel1Min and k[5] <= vel1Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 5])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp11_0.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                numberOfApp11_0.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt11_0.append(round(meanFoc, 2))

        for i in range(19, 24):
            workbook._sheets[2]['B' + str(i)] = ballastDt11_0[i - 19]

        ###SPEED 11.5  2 < WIND <3
        ballastDt11_3 = []
        numberOfApp11_3 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 2 and k[4] <= 3 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[i] and
                                 k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 2 and k[4] <= 3 and k[5] >= vel1Min and k[5] <= vel1Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp11_3.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                numberOfApp11_3.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt11_3.append(round(meanFoc, 2))

        for i in range(19, 24):
            workbook._sheets[2]['C' + str(i)] = ballastDt11_3[i - 19]

        ###SPEED 11.5  4 < WIND <5
        ballastDt11_5 = []
        numberOfApp11_5 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 4 and k[4] <= 5 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[i] and
                                 k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 4 and k[4] <= 5 and k[5] >= vel1Min and k[5] <= vel1Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp11_5.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                numberOfApp11_5.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt11_5.append(round(meanFoc, 2))

        for i in range(19, 24):
            workbook._sheets[2]['D' + str(i)] = ballastDt11_5[i - 19]

            ###SPEED 11.5  7 < WIND <8
        ballastDt11_8 = []
        numberOfApp11_8 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 6 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 6 and k[5] >= vel1Min and k[5] <= vel1Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(

                    [k for k in ballastDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp11_8.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt11_8.append(round(meanFoc, 2))

        for i in range(19, 24):
            workbook._sheets[2]['E' + str(i)] = ballastDt11_8[i - 19]

        #################################

        ###SPEED 12.5 WIND <1.5
        centralMean = np.mean(np.array([k for k in ballastDt if
                                        k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 8])
        centralArray = np.array([k for k in ballastDt if
                                 k[5] >= vel2Min and k[5] <= vel2Max and k[8] > 4])[:, 8]
        ballastDt12_0 = []
        numberOfApp12_0 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 0 and k[4] <= 1 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[
                                     i] and
                                 k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 0 and k[4] <= 1 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp12_0.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp12_0.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt12_0.append(round(meanFoc, 2))

        for i in range(29, 34):
            workbook._sheets[2]['B' + str(i)] = ballastDt12_0[i - 29]

        ###SPEED 11.5  2 < WIND <3
        ballastDt12_3 = []
        numberOfApp12_3 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 2 and k[4] <= 3 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 2 and k[4] <= 3 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp12_3.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp12_3.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt12_3.append(round(meanFoc, 2))

        for i in range(29, 34):
            workbook._sheets[2]['C' + str(i)] = ballastDt12_3[i - 29]

        ###SPEED 11.5  4 < WIND <5
        ballastDt12_5 = []
        numberOfApp12_5 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 4 and k[4] <= 5 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 4 and k[4] <= 5 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp12_5.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp12_5.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            ballastDt12_5.append(round(meanFoc, 2))

        for i in range(29, 34):
            workbook._sheets[2]['D' + str(i)] = ballastDt12_5[i - 29]

        ###SPEED 11.5  7 < WIND <8
        ballastDt12_8 = []
        numberOfApp12_8 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 6 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[i] and k[3] <= wind[
                                     i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 6 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp12_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp12_8.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt12_8.append(round(meanFoc, 2))

        for i in range(29, 34):
            workbook._sheets[2]['E' + str(i)] = ballastDt12_8[i - 29]

        #################################

        ###SPEED 13.5 WIND <1.5
        centralMean = np.array([k for k in ballastDt if
                                k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])
        if centralMean.__len__() > 0:
            centralMean = np.mean(np.array([k for k in ballastDt if
                                            k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])[:, 8])

            centralArray = np.array([k for k in ballastDt if
                                     k[5] >= vel3Min and k[5] <= vel3Max and k[8] > 4])
        else:
            centralMean = np.mean(np.array([k for k in ballastDt if
                                            k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 8]) + 1
            centralArray = np.array([k for k in ballastDt if
                                     k[5] >= vel2Min and k[5] <= vel2Max and k[8] > 4])[:, 8] + 1

        ballastDt13_0 = []
        numberOfApp13_0 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 0 and k[4] <= 1 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >=
                                 wind[i] and k[3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 0 and k[4] <= 1 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp13_0.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp13_0.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt13_0.append(round(meanFoc, 2))

        for i in range(39, 44):
            workbook._sheets[2]['B' + str(i)] = ballastDt13_0[i - 39]

        ###SPEED 13.5  2 < WIND <3
        ballastDt13_3 = []
        numberOfApp13_3 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 2 and k[4] <= 3 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 2 and k[4] <= 3 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp13_3.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp13_3.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt13_3.append(round(meanFoc, 2))

        for i in range(39, 44):
            workbook._sheets[2]['C' + str(i)] = ballastDt13_3[i - 39]

        ###SPEED 13.5  4 < WIND <5
        ballastDt13_5 = []
        numberOfApp13_5 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 4 and k[4] <= 5 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 4 and k[4] <= 5 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp13_5.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp13_5.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt13_5.append(round(meanFoc, 2))

        for i in range(39, 44):
            workbook._sheets[2]['D' + str(i)] = ballastDt13_5[i - 39]

        ###SPEED 13.5  7 < WIND <8
        ballastDt13_8 = []
        numberOfApp13_8 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ballastDt if
                                 k[4] >= 6 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ballastDt if
                                  k[4] >= 6 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array(
                [k for k in ballastDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])
            # tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ballastDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp13_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp13_8.append(arrayFoc.__len__() + centralArray.__len__())
            ballastDt13_8.append(round(meanFoc, 2))

        for i in range(39, 44):
            workbook._sheets[2]['E' + str(i)] = ballastDt13_8[i - 39]

        ####END OF BALLAST #############################################################
        ####END OF BALLAST #############################################################
        ####END OF BALLAST #############################################################

        ###WRITE BALLAST CONS TO FILE

        ##TREAT outliers / missing values for ballast values

        # workbook.save(filename=pathToexcel.split('.')[0] + '_1.' + pathToexcel.split('.')[1])
        # return
        #####################################################################################################################
        ##REPLACE NULL ZERO VALUES
        #####################################################################################################################
        values = [k for k in ballastDt10_0 if k != 0]
        length = values.__len__()
        numberOfApp10_0 = [numberOfApp10_0[i] for i in range(0, len(numberOfApp10_0)) if ballastDt10_0[i] > 0]
        for i in range(0, len(ballastDt10_0)):
            if length > 0:
                if length > 0:
                    if ballastDt10_0[i] == 0:
                        ##find items !=0
                        ballastDt10_0[i] = np.array(np.sum(
                            [numberOfApp10_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp10_0)
                    elif np.isnan(ballastDt10_0[i]):
                        ballastDt10_0[i] = np.array(np.sum(
                            [numberOfApp10_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp10_0)

        values = [k for k in ballastDt11_0 if k != 0]
        length = values.__len__()
        numberOfApp11_0 = [numberOfApp11_0[i] for i in range(0, len(numberOfApp11_0)) if ballastDt11_0[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt11_0)):
                if length > 0:

                    if ballastDt11_0[i] == 0:
                        ##find items !=0
                        ballastDt11_0[i] = np.array(np.sum(
                            [numberOfApp11_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp11_0)
                    elif np.isnan(ballastDt10_0[i]):
                        ballastDt11_0[i] = np.array(np.sum(
                            [numberOfApp11_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp11_0)

        values = [k for k in ballastDt12_0 if k != 0]
        length = values.__len__()
        numberOfApp12_0 = [numberOfApp12_0[i] for i in range(0, len(numberOfApp12_0)) if ballastDt12_0[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt12_0)):
                if length > 0:
                    if ballastDt12_0[i] == 0:
                        ##find items !=0
                        ballastDt12_0[i] = np.array(np.sum(
                            [numberOfApp12_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp12_0)
                    elif np.isnan(ballastDt10_0[i]):
                        ballastDt12_0[i] = np.array(np.sum(
                            [numberOfApp12_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp12_0)

        values = [k for k in ballastDt13_0 if k != 0]
        length = values.__len__()
        numberOfApp13_0 = [numberOfApp13_0[i] for i in range(0, len(numberOfApp13_0)) if ballastDt13_0[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt13_0)):
                if length > 0:
                    if ballastDt13_0[i] == 0:
                        ##find items !=0
                        ballastDt13_0[i] = np.array(np.sum(
                            [numberOfApp13_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp13_0)
                    elif np.isnan(ballastDt10_0[i]):
                        ballastDt13_0[i] = np.array(np.sum(
                            [numberOfApp13_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp13_0)

        values = [k for k in ballastDt10_3 if k != 0]
        length = values.__len__()
        numberOfApp10_3 = [numberOfApp10_3[i] for i in range(0, len(numberOfApp10_3)) if ballastDt10_3[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt10_3)):
                if length > 0:
                    if ballastDt10_3[i] == 0:
                        ##find items !=0
                        ballastDt10_3[i] = np.array(np.sum(
                            [numberOfApp10_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp10_3)
                    elif np.isnan(ballastDt10_3[i]):
                        ballastDt10_0[i] = np.array(np.sum(
                            [numberOfApp10_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp10_3)

        values = [k for k in ballastDt11_3 if k != 0]
        length = values.__len__()
        if values.__len__() > 0:
            numberOfApp11_3 = [numberOfApp11_3[i] for i in range(0, len(numberOfApp11_3)) if ballastDt11_3[i] > 0]
            for i in range(0, len(ballastDt11_3)):
                if length > 0:
                    if ballastDt11_3[i] == 0:
                        ##find items !=0
                        ballastDt11_3[i] = np.array(np.sum(
                            [numberOfApp11_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp11_3)
                    elif np.isnan(ballastDt11_3[i]):
                        ballastDt11_3[i] = np.array(np.sum(
                            [numberOfApp11_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp11_3)

        values = [k for k in ballastDt12_3 if k != 0]
        length = values.__len__()
        numberOfApp12_3 = [numberOfApp12_3[i] for i in range(0, len(numberOfApp12_3)) if ballastDt12_3[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt12_3)):
                if length > 0:
                    if ballastDt12_3[i] == 0:
                        ##find items !=0
                        ballastDt12_3[i] = np.array(np.sum(
                            [numberOfApp12_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp12_3)
                    elif np.isnan(ballastDt12_3[i]):
                        ballastDt12_3[i] = np.array(np.sum(
                            [numberOfApp12_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp12_3)

        values = [k for k in ballastDt13_3 if k != 0]
        length = values.__len__()
        numberOfApp13_3 = [numberOfApp13_3[i] for i in range(0, len(numberOfApp13_3)) if ballastDt13_3[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt13_3)):
                if length > 0:
                    if ballastDt13_3[i] == 0:
                        ##find items !=0
                        ballastDt13_3[i] = np.array(np.sum(
                            [numberOfApp13_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp13_3)
                    elif np.isnan(ballastDt13_3[i]):
                        ballastDt12_3[i] = np.array(np.sum(
                            [numberOfApp13_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp13_3)

        values = [k for k in ballastDt10_5 if k != 0]
        length = values.__len__()
        numberOfApp10_5 = [numberOfApp10_5[i] for i in range(0, len(numberOfApp10_5)) if ballastDt10_5[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt10_5)):
                if length > 0:
                    if ballastDt10_5[i] == 0:
                        ##find items !=0
                        ballastDt10_5[i] = np.array(np.sum(
                            [numberOfApp10_5[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp10_5)
                    elif np.isnan(ballastDt12_3[i]):
                        ballastDt12_3[i] = np.array(np.sum(
                            [numberOfApp10_5[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp10_5)

        values = [k for k in ballastDt11_5 if k != 0]
        length = values.__len__()
        numberOfApp11_5 = [numberOfApp11_5[i] for i in range(0, len(numberOfApp11_5)) if ballastDt11_5[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt11_5)):
                if length > 0:
                    if ballastDt11_5[i] == 0:
                        ##find items !=0
                        ballastDt11_5[i] = np.array(np.sum(
                            [numberOfApp11_5[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp11_5)
                    elif np.isnan(ballastDt11_5[i]):
                        ballastDt11_5[i] = np.array(np.sum(
                            [numberOfApp10_5[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp11_5)

        values = [k for k in ballastDt12_5 if k != 0]
        length = values.__len__()
        numberOfApp12_5 = [numberOfApp12_5[i] for i in range(0, len(numberOfApp12_5)) if ballastDt12_5[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt12_5)):
                if length > 0:
                    if ballastDt12_5[i] == 0:
                        ##find items !=0
                        ballastDt12_5[i] = np.array(np.sum(
                            [numberOfApp12_5[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp12_5)
                    elif np.isnan(ballastDt12_5[i]):
                        ballastDt12_5[i] = np.array(np.sum(
                            [numberOfApp12_5[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp12_5)

        values = [k for k in ballastDt13_5 if k != 0]
        length = values.__len__()
        numberOfApp13_5 = [numberOfApp13_5[i] for i in range(0, len(numberOfApp13_5)) if ballastDt13_5[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt13_5)):
                if length > 0:
                    if ballastDt13_5[i] == 0:
                        ##find items !=0
                        ballastDt13_5[i] = np.array(np.sum(
                            [numberOfApp13_5[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp13_5)
                    elif np.isnan(ballastDt13_5[i]):
                        ballastDt13_5[i] = np.array(np.sum(
                            [numberOfApp13_5[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp13_5)

        values = [k for k in ballastDt10_8 if k != 0]
        length = values.__len__()
        numberOfApp10_8 = [numberOfApp10_8[i] for i in range(0, len(numberOfApp10_8)) if ballastDt10_8[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt10_8)):
                if length > 0:
                    if ballastDt10_8[i] == 0:
                        ##find items !=0
                        ballastDt10_8[i] = np.array(np.sum(
                            [numberOfApp10_8[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp10_8)
                    elif np.isnan(ballastDt10_8[i]):
                        ballastDt10_8[i] = np.array(np.sum(
                            [numberOfApp10_8[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp10_8)

        values = [k for k in ballastDt11_8 if k != 0]
        length = values.__len__()
        numberOfApp11_8 = [numberOfApp11_8[i] for i in range(0, len(numberOfApp11_8)) if ballastDt11_8[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt11_8)):
                if length > 0:
                    if ballastDt11_8[i] == 0:
                        ##find items !=0
                        ballastDt11_8[i] = np.array(np.sum(
                            [numberOfApp11_8[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp11_8)
                    elif np.isnan(ballastDt11_8[i]):
                        ballastDt11_8[i] = np.array(np.sum(
                            [numberOfApp11_8[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp11_8)

        values = [k for k in ballastDt12_8 if k != 0]
        length = values.__len__()
        numberOfApp12_8 = [numberOfApp12_8[i] for i in range(0, len(numberOfApp12_8)) if ballastDt12_8[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt12_8)):
                if length > 0:
                    if ballastDt12_8[i] == 0:
                        ##find items !=0
                        ballastDt12_8[i] = np.array(np.sum(
                            [numberOfApp12_8[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp12_8)
                    elif np.isnan(ballastDt12_8[i]):
                        ballastDt12_8[i] = np.array(np.sum(
                            [numberOfApp12_8[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp12_8)

        values = [k for k in ballastDt13_8 if k != 0]
        length = values.__len__()
        numberOfApp13_8 = [numberOfApp13_8[i] for i in range(0, len(numberOfApp13_8)) if ballastDt13_8[i] > 0]
        if values.__len__() > 0:
            for i in range(0, len(ballastDt13_8)):
                if length > 0:
                    if ballastDt13_8[i] == 0:
                        ##find items !=0
                        ballastDt13_8[i] = np.array(np.sum(
                            [numberOfApp13_8[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp13_8)
                    elif np.isnan(ballastDt13_8[i]):
                        ballastDt13_8[i] = np.array(np.sum(
                            [numberOfApp13_8[i] * values[i] for i in range(0, len(values))])) / np.sum(
                            numberOfApp13_8)

        ################################################################################################################
        if (np.array(ballastDt10_0) == 0).all() and (np.array(ballastDt10_3) == 0).all() and (
                np.array(ballastDt10_5) == 0).all() and (np.array(ballastDt10_8) == 0).all():
            ballastDt10_0 = np.array(ballastDt11_0) - 2
            ballastDt10_3 = np.array(ballastDt11_3) - 2
            ballastDt10_5 = np.array(ballastDt11_5) - 2
            ballastDt10_8 = np.array(ballastDt11_8) - 2

        values = [k for k in ballastDt10_0 if k != 0]
        if values.__len__() == 0 or (np.array(ballastDt10_0) <= 0).all():
            ballastDt10_0 = np.array(ballastDt10_3) - 1

        values = [k for k in ballastDt11_0 if k != 0]
        if values.__len__() == 0 or (np.array(ballastDt11_0) <= 0).all():
            ballastDt11_0 = np.array(ballastDt11_3) - 1
        values = [k for k in ballastDt12_0 if k != 0]

        if values.__len__() == 0 or (np.array(ballastDt12_0) <= 0).all():
            ballastDt12_0 = np.array(ballastDt12_3) - 1

        values = [k for k in ballastDt13_0 if k != 0]
        if values.__len__() == 0 or (np.array(ballastDt13_0) <= 0).all():
            ballastDt13_0 = np.array(ballastDt13_3) - 1
        #####################################################################################################################
        #####################################################################################################################

        #####################################################################################################################
        for i in range(0, len(ballastDt10_0)):

            if (ballastDt10_0[i] >= ballastDt11_0[i]) and ballastDt11_0[i] > 0:
                while (ballastDt10_0[i] >= ballastDt11_0[i]):
                    ballastDt11_0[i] = ballastDt11_0[i] + 0.1 * ballastDt11_0[i]

        for i in range(0, len(ballastDt11_0)):

            if (ballastDt11_0[i] >= ballastDt12_0[i]) and ballastDt12_0[i] > 0:
                while (ballastDt11_0[i] >= ballastDt12_0[i]):
                    ballastDt12_0[i] = ballastDt12_0[i] + 0.1 * ballastDt12_0[i]

        for i in range(0, len(ballastDt12_0)):

            if (ballastDt12_0[i] >= ballastDt13_0[i]) and ballastDt13_0[i] > 0:
                while (ballastDt12_0[i] >= ballastDt13_0[i]):
                    ballastDt13_0[i] = ballastDt13_0[i] + 0.1 * ballastDt13_0[i]

        #####################################################################################################################
        for i in range(0, len(ballastDt10_3)):

            if (ballastDt10_0[i] >= ballastDt10_3[i]):
                while (ballastDt10_0[i] >= ballastDt10_3[i]):
                    if ballastDt10_3[i] == 0:
                        ballastDt10_3[i] = ballastDt10_0[i] + 0.1 * ballastDt10_0[i]
                    else:
                        ballastDt10_3[i] = ballastDt10_3[i] + 0.1 * ballastDt10_3[i]

        for i in range(0, len(ballastDt10_5)):

            if (ballastDt10_3[i] >= ballastDt10_5[i]):
                while (ballastDt10_3[i] >= ballastDt10_5[i]):
                    if ballastDt10_5[i] == 0:
                        ballastDt10_5[i] = ballastDt10_3[i] + 0.1 * ballastDt10_3[i]
                    else:
                        ballastDt10_5[i] = ballastDt10_5[i] + 0.1 * ballastDt10_5[i]

        for i in range(0, len(ballastDt10_8)):

            if (ballastDt10_5[i] >= ballastDt10_8[i]):
                while (ballastDt10_5[i] >= ballastDt10_8[i]):
                    if ballastDt10_8[i] == 0:
                        ballastDt10_8[i] = ballastDt10_5[i] + 0.1 * ballastDt10_5[i]
                    else:
                        ballastDt10_8[i] = ballastDt10_8[i] + 0.1 * ballastDt10_8[i]
        #####################################################################################################################

        for i in range(0, len(ballastDt11_3)):

            if (ballastDt11_0[i] >= ballastDt11_3[i]):
                while (ballastDt11_0[i] >= ballastDt11_3[i]):
                    if ballastDt11_3[i] == 0:
                        ballastDt11_3[i] = ballastDt11_0[i] + 0.1 * ballastDt11_0[i]
                    else:
                        ballastDt11_3[i] = ballastDt11_3[i] + 0.1 * ballastDt11_3[i]

        for i in range(0, len(ballastDt11_5)):

            if (ballastDt11_3[i] >= ballastDt11_5[i]):
                while (ballastDt11_3[i] >= ballastDt11_5[i]):
                    if ballastDt11_5[i] == 0:
                        ballastDt11_5[i] = ballastDt11_3[i] + 0.1 * ballastDt11_3[i]
                    else:
                        ballastDt11_5[i] = ballastDt11_5[i] + 0.1 * ballastDt11_5[i]

        for i in range(0, len(ballastDt11_8)):

            if (ballastDt11_5[i] >= ballastDt11_8[i]):
                while (ballastDt11_5[i] > ballastDt11_8[i]):
                    if ballastDt11_8[i] == 0:
                        ballastDt11_8[i] = ballastDt11_5[i] + 0.1 * ballastDt11_5[i]
                    else:
                        ballastDt11_8[i] = ballastDt11_8[i] + 0.1 * ballastDt11_8[i]

        #####################################################################################################################

        for i in range(0, len(ballastDt10_3)):

            if (ballastDt10_3[i] > ballastDt11_3[i]) and ballastDt11_3[i] > 0:
                while (ballastDt10_3[i] > ballastDt11_3[i]):
                    ballastDt11_3[i] = ballastDt11_3[i] + 0.1 * ballastDt11_3[i]

        for i in range(0, len(ballastDt10_5)):

            if (ballastDt10_5[i] > ballastDt11_5[i]) and ballastDt11_5[i] > 0:
                while (ballastDt10_5[i] > ballastDt11_5[i]):
                    ballastDt11_5[i] = ballastDt11_5[i] + 0.1 * ballastDt11_5[i]

        for i in range(0, len(ballastDt10_5)):

            if (ballastDt10_8[i] > ballastDt11_8[i]) and ballastDt11_8[i] > 0:
                while (ballastDt10_8[i] > ballastDt11_8[i]):
                    ballastDt11_8[i] = ballastDt11_8[i] + 0.1 * ballastDt11_8[i]
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #####################################################################################################################

        for i in range(0, len(ballastDt12_3)):

            if (ballastDt12_0[i] > ballastDt12_3[i]):
                while (ballastDt12_0[i] > ballastDt12_3[i]):
                    if ballastDt12_3[i] == 0:
                        ballastDt12_3[i] = ballastDt12_0[i] + 0.1 * ballastDt12_0[i]
                    else:
                        ballastDt12_3[i] = ballastDt12_3[i] + 0.1 * ballastDt12_3[i]

        for i in range(0, len(ballastDt12_5)):

            if (ballastDt12_3[i] > ballastDt12_5[i]):
                while (ballastDt12_3[i] > ballastDt12_5[i]):
                    if ballastDt12_5[i] == 0:
                        ballastDt12_5[i] = ballastDt12_3[i] + 0.1 * ballastDt12_3[i]
                    else:
                        ballastDt12_5[i] = ballastDt12_5[i] + 0.1 * ballastDt12_5[i]

        for i in range(0, len(ballastDt12_8)):

            if (ballastDt12_5[i] > ballastDt12_8[i]):
                while (ballastDt12_5[i] > ballastDt12_8[i]):
                    if ballastDt12_8[i] == 0:
                        ballastDt12_8[i] = ballastDt12_5[i] + 0.1 * ballastDt12_5[i]
                    else:
                        ballastDt12_8[i] = ballastDt12_8[i] + 0.1 * ballastDt12_8[i]

        #####################################################################################################################
        for i in range(0, len(ballastDt12_3)):

            if (ballastDt11_3[i] > ballastDt12_3[i]) and ballastDt12_3[i] > 0:
                while (ballastDt11_3[i] > ballastDt12_3[i]):
                    ballastDt12_3[i] = ballastDt12_3[i] + 0.1 * ballastDt12_3[i]

        for i in range(0, len(ballastDt12_5)):

            if (ballastDt11_5[i] > ballastDt12_5[i]) and ballastDt12_5[i] > 0:
                while (ballastDt11_5[i] > ballastDt12_5[i]):
                    ballastDt12_5[i] = ballastDt12_5[i] + 0.1 * ballastDt12_5[i]

        for i in range(0, len(ballastDt12_8)):

            if (ballastDt11_8[i] > ballastDt12_8[i]) and ballastDt12_8[i] > 0:
                while (ballastDt11_8[i] > ballastDt12_8[i]):
                    ballastDt12_8[i] = ballastDt12_8[i] + 0.1 * ballastDt12_8[i]
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #####################################################################################################################

        for i in range(0, len(ballastDt13_3)):

            if (ballastDt13_0[i] > ballastDt13_3[i]):
                while (ballastDt13_0[i] > ballastDt13_3[i]):
                    if ballastDt13_3[i] == 0:
                        ballastDt13_3[i] = ballastDt12_0[i] + 0.1 * ballastDt12_0[i]
                    else:
                        ballastDt13_3[i] = ballastDt13_3[i] + 0.1 * ballastDt13_3[i]

        for i in range(0, len(ballastDt13_5)):

            if (ballastDt13_3[i] > ballastDt13_5[i]):
                while (ballastDt13_3[i] > ballastDt13_5[i]):
                    if ballastDt13_5[i] == 0:
                        ballastDt13_5[i] = ballastDt12_3[i] + 0.1 * ballastDt12_3[i]
                    else:
                        ballastDt13_5[i] = ballastDt13_5[i] + 0.1 * ballastDt13_5[i]

        for i in range(0, len(ballastDt13_8)):

            if (ballastDt13_5[i] > ballastDt13_8[i]):
                while (ballastDt13_5[i] > ballastDt13_8[i]):
                    if ballastDt13_8[i] == 0:
                        ballastDt13_8[i] = ballastDt12_5[i] + 0.1 * ballastDt12_5[i]
                    else:
                        ballastDt13_8[i] = ballastDt13_8[i] + 0.1 * ballastDt13_8[i]

        #####################################################################################################################

        for i in range(0, len(ballastDt13_3)):

            if (ballastDt12_3[i] > ballastDt13_3[i]) and ballastDt13_3[i] > 0:
                while (ballastDt12_3[i] > ballastDt13_3[i]):
                    ballastDt13_3[i] = ballastDt13_3[i] + 0.1 * ballastDt13_3[i]

        for i in range(0, len(ballastDt13_5)):

            if (ballastDt12_5[i] > ballastDt13_5[i]) and ballastDt13_5[i] > 0:
                while (ballastDt12_5[i] > ballastDt13_5[i]):
                    ballastDt13_5[i] = ballastDt13_5[i] + 0.1 * ballastDt13_5[i]

        for i in range(0, len(ballastDt13_8)):
            if (ballastDt12_8[i] > ballastDt13_8[i]) and ballastDt13_8[i] > 0:
                while (ballastDt12_8[i] > ballastDt13_8[i]):
                    ballastDt13_8[i] = ballastDt13_8[i] + 0.1 * ballastDt13_8[i]
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (np.array(ballastDt10_3) == 0).all():
            ballastDt10_3 = np.array(ballastDt10_0) + 1.5

        if (np.array(ballastDt10_5) == 0).all():
            ballastDt10_5 = np.array(ballastDt10_3) + 1.5

        if (np.array(ballastDt10_8) == 0).all():
            ballastDt10_8 = np.array(ballastDt10_5) + 1.5

        if (np.array(ballastDt11_0) == 0).all():
            ballastDt11_0 = np.array(ballastDt10_0) + 1

        if (np.array(ballastDt11_3) == 0).all():
            ballastDt11_3 = np.array(ballastDt11_0) + 1.5

        if (np.array(ballastDt11_5) == 0).all():
            ballastDt11_5 = np.array(ballastDt11_3) + 1.5

        if (np.array(ballastDt11_8) == 0).all():
            ballastDt11_8 = np.array(ballastDt11_5) + 1.5

        if (np.array(ballastDt12_0) == 0).all() or (np.array(ballastDt12_0) < 0).any():
            ballastDt12_0 = np.array(ballastDt11_0) + 1

        if (np.array(ballastDt12_3) == 0).all():
            ballastDt12_3 = np.array(ballastDt12_0) + 1.5
        for i in range(0, len(ballastDt12_3)):
            if ballastDt12_3[i] == inf:
                ballastDt12_3[i] = 0
        if (np.array(ballastDt12_5) == 0).all():
            ballastDt12_5 = np.array(ballastDt12_3) + 1.5

        if (np.array(ballastDt12_8) == 0).all():
            ballastDt12_8 = np.array(ballastDt12_5) + 1.5

        if (np.array(ballastDt13_0) == 0).all() or (np.array(ballastDt13_0) < 0).any():
            ballastDt13_0 = np.array(ballastDt12_0) + 1

        if (np.array(ballastDt13_3) == 0).all():
            ballastDt13_3 = np.array(ballastDt13_0) + 1.5

        if (np.array(ballastDt13_5) == 0).all():
            ballastDt13_5 = np.array(ballastDt13_3) + 1.5

        if (np.array(ballastDt13_8) == 0).all():
            ballastDt13_8 = np.array(ballastDt13_5) + 1.5
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (np.array(ballastDt10_0) == 0).all():
            ballastDt10_0 = np.array(ballastDt10_3) - 1 if (np.array(ballastDt10_3) != 0).any() else np.array(
                ballastDt10_5) - 2
        # if (np.array(ballastDt10_0)!=0).all()
        for i in range(9, 14):
            ballastDt10_0[0] = ballastDt10_0[4] + 1 if ballastDt10_0[0] <= ballastDt10_0[4] else ballastDt10_0[0]
            ballastDt10_0[2] = ballastDt10_0[3] + 1 if ballastDt10_0[2] <= ballastDt10_0[3] else ballastDt10_0[2]
            workbook._sheets[2]['B' + str(i)] = round(ballastDt10_0[i - 9], 2)

        ##TREAT outliers / missing values for ballastt values
        if (np.array(ballastDt10_3) == 0).all():
            ballastDt10_3 = np.array(ballastDt10_5) - 1 if (np.array(ballastDt10_5) != 0).any() else np.array(
                ballastDt10_8) - 3
        for i in range(9, 14):
            ballastDt10_3[0] = ballastDt10_3[4] + 1 if ballastDt10_3[0] <= ballastDt10_3[4] else ballastDt10_3[0]
            ballastDt10_3[2] = ballastDt10_3[3] + 1 if ballastDt10_3[2] <= ballastDt10_3[3] else ballastDt10_3[2]
            workbook._sheets[2]['C' + str(i)] = round(ballastDt10_3[i - 9], 2)

            ##TREAT outliers / missing values for ballastt values
        if (np.array(ballastDt10_5) == 0).all():
            ballastDt10_5 = np.array(ballastDt10_8) - 1.5 if (np.array(ballastDt10_8) != 0).any() else np.array(
                ballastDt10_3) + 1
        for i in range(9, 14):
            ballastDt10_5[0] = ballastDt10_5[4] + 1 if ballastDt10_5[0] <= ballastDt10_5[4] else ballastDt10_5[0]
            ballastDt10_5[2] = ballastDt10_5[3] + 1 if ballastDt10_5[2] <= ballastDt10_5[3] else ballastDt10_5[2]
            workbook._sheets[2]['D' + str(i)] = round(ballastDt10_5[i - 9], 2)

        ##############################################################################################################################
        ##FIX SIDE  / AGAINST
        if (ballastDt10_0[0] <= ballastDt10_0[2]):
            while (ballastDt10_0[0] <= ballastDt10_0[2]):
                ballastDt10_0[0] = ballastDt10_0[0] + 0.1 * ballastDt10_0[0]

        if (ballastDt10_0[1] <= ballastDt10_0[3]):
            while (ballastDt10_0[1] <= ballastDt10_0[3]):
                ballastDt10_0[1] = ballastDt10_0[1] + 0.1 * ballastDt10_0[1]

        if (ballastDt10_3[0] <= ballastDt10_3[2]):
            while (ballastDt10_3[0] <= ballastDt10_3[2]):
                ballastDt10_3[0] = ballastDt10_3[0] + 0.1 * ballastDt10_3[0]

        if (ballastDt10_3[1] <= ballastDt10_3[3]):
            while (ballastDt10_3[1] <= ballastDt10_3[3]):
                ballastDt10_3[1] = ballastDt10_3[1] + 0.1 * ballastDt10_3[1]

        if (ballastDt10_5[0] <= ballastDt10_5[2]):
            while (ballastDt10_5[0] <= ballastDt10_5[2]):
                ballastDt10_5[0] = ballastDt10_5[0] + 0.1 * ballastDt10_5[0]

        if (ballastDt10_5[1] <= ballastDt10_5[3]):
            while (ballastDt10_5[1] <= ballastDt10_5[3]):
                ballastDt10_5[1] = ballastDt10_5[1] + 0.1 * ballastDt10_5[1]

        if (ballastDt10_8[0] <= ballastDt10_8[2]):
            while (ballastDt10_8[0] <= ballastDt10_8[2]):
                ballastDt10_8[0] = ballastDt10_8[0] + 0.1 * ballastDt10_8[0]

        if (ballastDt10_8[1] <= ballastDt10_8[3]):
            while (ballastDt10_8[1] <= ballastDt10_8[3]):
                ballastDt10_8[1] = ballastDt10_8[1] + 0.1 * ballastDt10_8[1]

        if (ballastDt11_0[0] <= ballastDt11_0[2]):
            while (ballastDt11_0[0] <= ballastDt11_0[2]):
                ballastDt11_0[0] = ballastDt11_0[0] + 0.1 * ballastDt11_0[0]

        if (ballastDt11_0[1] <= ballastDt11_0[3]):
            while (ballastDt11_0[1] <= ballastDt11_0[3]):
                ballastDt11_0[1] = ballastDt11_0[1] + 0.1 * ballastDt11_0[1]

        if (ballastDt11_3[0] <= ballastDt11_3[2]):
            while (ballastDt11_3[0] <= ballastDt11_3[2]):
                ballastDt11_3[0] = ballastDt11_3[0] + 0.1 * ballastDt11_3[0]

        if (ballastDt11_3[1] <= ballastDt11_3[3]):
            while (ballastDt11_3[1] <= ballastDt11_3[3]):
                ballastDt11_3[1] = ballastDt11_3[1] + 0.1 * ballastDt11_3[1]

        if (ballastDt11_5[0] <= ballastDt11_5[2]):
            while (ballastDt11_5[0] <= ballastDt11_5[2]):
                ballastDt11_5[0] = ballastDt11_5[0] + 0.1 * ballastDt11_5[0]

        if (ballastDt11_5[1] <= ballastDt11_5[3]):
            while (ballastDt11_5[1] <= ballastDt11_5[3]):
                ballastDt11_5[1] = ballastDt11_5[1] + 0.1 * ballastDt11_5[1]

        if (ballastDt11_8[0] <= ballastDt11_8[2]):
            while (ballastDt11_8[0] <= ballastDt11_8[2]):
                ballastDt11_8[0] = ballastDt11_8[0] + 0.1 * ballastDt11_8[0]

        if (ballastDt11_8[1] <= ballastDt11_8[3]):
            while (ballastDt11_8[1] <= ballastDt11_8[3]):
                ballastDt11_8[1] = ballastDt11_8[1] + 0.1 * ballastDt11_8[1]

        if (ballastDt12_0[0] <= ballastDt12_0[2]):
            while (ballastDt12_0[0] <= ballastDt12_0[2]):
                ballastDt12_0[0] = ballastDt12_0[0] + 0.1 * ballastDt12_0[0]

        if (ballastDt12_0[1] <= ballastDt12_0[3]):
            while (ballastDt12_0[1] <= ballastDt12_0[3]):
                ballastDt12_0[1] = ballastDt12_0[1] + 0.1 * ballastDt12_0[1]

        if (ballastDt12_3[0] <= ballastDt12_3[2]):
            while (ballastDt12_3[0] <= ballastDt12_3[2]):
                ballastDt12_3[0] = ballastDt12_3[0] + 0.1 * ballastDt12_3[0]

        if (ballastDt12_3[1] <= ballastDt12_3[3]):
            while (ballastDt12_3[1] <= ballastDt12_3[3]):
                ballastDt12_3[1] = ballastDt12_3[1] + 0.1 * ballastDt12_3[1]

        if (ballastDt12_5[0] <= ballastDt12_5[2]):
            while (ballastDt12_5[0] <= ballastDt12_5[2]):
                ballastDt12_5[0] = ballastDt12_5[0] + 0.1 * ballastDt12_5[0]

        if (ballastDt12_5[1] <= ballastDt12_5[3]):
            while (ballastDt12_5[1] <= ballastDt12_5[3]):
                ballastDt12_5[1] = ballastDt12_5[1] + 0.1 * ballastDt12_5[1]

        if (ballastDt12_8[0] <= ballastDt12_8[2]):
            while (ballastDt12_8[0] <= ballastDt12_8[2]):
                ballastDt12_8[0] = ballastDt12_8[0] + 0.1 * ballastDt12_8[0]

        if (ballastDt12_8[1] <= ballastDt12_8[3]):
            while (ballastDt12_8[1] <= ballastDt12_8[3]):
                ballastDt12_8[1] = ballastDt12_8[1] + 0.1 * ballastDt12_8[1]

        if (ballastDt13_0[0] <= ballastDt13_0[2]):
            while (ballastDt13_0[0] <= ballastDt13_0[2]):
                ballastDt13_0[0] = ballastDt13_0[0] + 0.1 * ballastDt13_0[0]

        if (ballastDt13_0[1] <= ballastDt13_0[3]):
            while (ballastDt13_0[1] <= ballastDt13_0[3]):
                ballastDt13_0[1] = ballastDt13_0[1] + 0.1 * ballastDt13_0[1]

        if (ballastDt13_3[0] <= ballastDt13_3[2]):
            while (ballastDt13_3[0] <= ballastDt13_3[2]):
                ballastDt13_3[0] = ballastDt13_3[0] + 0.1 * ballastDt13_3[0]

        if (ballastDt13_3[1] <= ballastDt13_3[3]):
            while (ballastDt13_3[1] <= ballastDt13_3[3]):
                ballastDt13_3[1] = ballastDt13_3[1] + 0.1 * ballastDt13_3[1]

        if (ballastDt13_5[0] <= ballastDt13_5[2]):
            while (ballastDt13_5[0] <= ballastDt13_5[2]):
                ballastDt13_5[0] = ballastDt13_5[0] + 0.1 * ballastDt13_5[0]

        if (ballastDt13_5[1] <= ballastDt13_5[3]):
            while (ballastDt13_5[1] <= ballastDt13_5[3]):
                ballastDt13_5[1] = ballastDt13_5[1] + 0.1 * ballastDt13_5[1]

        if (ballastDt13_8[0] <= ballastDt13_8[2]):
            while (ballastDt13_8[0] <= ballastDt13_8[2]):
                ballastDt13_8[0] = ballastDt13_8[0] + 0.1 * ballastDt13_8[0]

        if (ballastDt13_8[1] <= ballastDt13_8[3]):
            while (ballastDt13_8[1] <= ballastDt13_8[3]):
                ballastDt13_8[1] = ballastDt13_8[1] + 0.1 * ballastDt13_8[1]
        ##FIX SIDE  / AGAINST##FIX SIDE  / AGAINST##FIX SIDE  / AGAINST##FIX SIDE  / AGAINST##FIX SIDE  / AGAINST##FIX SIDE  / AGAINST

        ##TREAT outliers / missing values for ballastt values
        if (np.array(ballastDt10_8) == 0).all():
            ballastDt10_8 = np.array(ballastDt10_5) + 2 if (np.array(ballastDt10_5) != 0).any() else np.array(
                ballastDt10_3) + 3
        for i in range(9, 14):
            ballastDt10_8[0] = ballastDt10_8[4] + 1 if ballastDt10_8[0] <= ballastDt10_8[4] else ballastDt10_8[0]
            ballastDt10_8[2] = ballastDt10_8[2] + 1 if ballastDt10_8[2] <= ballastDt10_8[3] else ballastDt10_8[2]
            workbook._sheets[2]['E' + str(i)] = round(ballastDt10_8[i - 9], 2)

            ####################################################################################################

            # values = [k for k in ballastDt11_0 if k != 0]
            # length = values.__len__()

        for i in range(19, 24):
            ballastDt11_0[0] = ballastDt11_0[4] + 1 if ballastDt11_0[0] <= ballastDt11_0[4] else ballastDt11_0[0]
            ballastDt11_0[2] = ballastDt11_0[3] + 1 if ballastDt11_0[2] <= ballastDt11_0[3] else ballastDt11_0[2]
            workbook._sheets[2]['B' + str(i)] = round(ballastDt11_0[i - 19], 2)

            ##TREAT outliers / missing values for ballastt values

        for i in range(19, 24):
            ballastDt11_3[0] = ballastDt11_3[4] + 1 if ballastDt11_3[0] <= ballastDt11_3[4] else ballastDt11_3[0]
            ballastDt11_3[2] = ballastDt11_3[3] + 1 if ballastDt11_3[2] <= ballastDt11_3[3] else ballastDt11_3[2]
            workbook._sheets[2]['C' + str(i)] = round(ballastDt11_3[i - 19], 2)

            ##TREAT outliers / missing values for ballastt values

        for i in range(19, 24):
            ballastDt11_5[0] = ballastDt11_5[4] + 1 if ballastDt11_5[0] <= ballastDt11_5[4] else ballastDt11_5[0]
            ballastDt11_5[2] = ballastDt11_5[3] + 1 if ballastDt11_5[2] <= ballastDt11_5[3] else ballastDt11_5[2]
            workbook._sheets[2]['D' + str(i)] = round(ballastDt11_5[i - 19], 2)

            ##TREAT outliers / missing values for ballastt values

        for i in range(19, 24):
            ballastDt11_8[0] = ballastDt11_8[4] + 1 if ballastDt11_8[0] <= ballastDt11_8[4] else ballastDt11_8[0]
            ballastDt11_8[2] = ballastDt11_8[3] + 1 if ballastDt11_8[2] <= ballastDt11_8[3] else ballastDt11_8[2]
            workbook._sheets[2]['E' + str(i)] = round(ballastDt11_8[i - 19], 2)

            ####################################################################################################

        for i in range(29, 34):
            ballastDt12_0[0] = ballastDt12_0[4] + 1 if ballastDt12_0[0] <= ballastDt12_0[4] else ballastDt12_0[0]
            ballastDt12_0[2] = ballastDt12_0[3] + 1 if ballastDt12_0[2] <= ballastDt12_0[3] else ballastDt12_0[2]
            workbook._sheets[2]['B' + str(i)] = round(ballastDt12_0[i - 29], 2)

            ##TREAT outliers / missing values for ballastt values

        for i in range(29, 34):
            ballastDt12_3[0] = ballastDt12_3[4] + 1 if ballastDt12_3[0] <= ballastDt12_3[4] else ballastDt12_3[0]
            ballastDt12_3[2] = ballastDt12_3[3] + 1 if ballastDt12_3[2] <= ballastDt12_3[3] else ballastDt12_3[2]
            workbook._sheets[2]['C' + str(i)] = round(ballastDt12_3[i - 29], 2)

            ##TREAT outliers / missing values for ballastt values

        for i in range(29, 34):
            ballastDt12_5[0] = ballastDt12_5[4] + 1 if ballastDt12_5[0] <= ballastDt12_5[4] else ballastDt12_5[0]
            ballastDt12_5[2] = ballastDt12_5[3] + 1 if ballastDt12_5[2] <= ballastDt12_5[3] else ballastDt12_5[2]
            workbook._sheets[2]['D' + str(i)] = round(ballastDt12_5[i - 29], 2)

            ##TREAT outliers / missing values for ballastt values

        for i in range(29, 34):
            ballastDt12_8[0] = ballastDt12_8[4] + 1 if ballastDt12_8[0] <= ballastDt12_8[4] else ballastDt12_8[0]
            ballastDt12_8[2] = ballastDt12_8[3] + 1 if ballastDt12_8[2] <= ballastDt12_8[3] else ballastDt12_8[2]
            workbook._sheets[2]['E' + str(i)] = round(ballastDt12_8[i - 29], 2)

            ################################################################################################################
        for i in range(39, 44):
            ballastDt13_0[0] = ballastDt13_0[4] + 1 if ballastDt13_0[0] <= ballastDt13_0[4] else ballastDt13_0[0]
            ballastDt13_0[2] = ballastDt13_0[3] + 1 if ballastDt13_0[2] <= ballastDt13_0[3] else ballastDt13_0[2]
            workbook._sheets[2]['B' + str(i)] = round(ballastDt13_0[i - 39], 2)

            ##TREAT outliers / missing values for ballastt values

        for i in range(39, 44):
            ballastDt13_3[0] = ballastDt13_3[4] + 1 if ballastDt13_3[0] <= ballastDt13_3[4] else ballastDt13_3[0]
            ballastDt13_3[2] = ballastDt13_3[3] + 1 if ballastDt13_3[2] <= ballastDt13_3[3] else ballastDt13_3[2]
            workbook._sheets[2]['C' + str(i)] = round(ballastDt13_3[i - 39], 2)

            ##TREAT outliers / missing values for ballastt values

        for i in range(39, 44):
            ballastDt13_5[0] = ballastDt13_5[4] + 1 if ballastDt13_5[0] <= ballastDt13_5[4] else ballastDt13_5[0]
            ballastDt13_5[2] = ballastDt13_5[3] + 1 if ballastDt13_5[2] <= ballastDt13_5[3] else ballastDt13_5[2]
            workbook._sheets[2]['D' + str(i)] = round(ballastDt13_5[i - 39], 2)

            ##TREAT outliers / missing values for ballastt values

        for i in range(39, 44):
            ballastDt13_8[0] = ballastDt13_8[4] + 1 if ballastDt13_8[0] <= ballastDt13_8[4] else ballastDt13_8[0]
            ballastDt13_8[2] = ballastDt13_8[3] + 1 if ballastDt13_8[2] <= ballastDt13_8[3] else ballastDt13_8[2]
            workbook._sheets[2]['E' + str(i)] = round(ballastDt13_8[i - 39], 2)

        ballastCons = list(itertools.chain(ballastDt10_0, ballastDt10_3, ballastDt10_5, ballastDt10_8,
                                           ballastDt11_0, ballastDt11_3, ballastDt11_5, ballastDt10_8,
                                           ballastDt12_0, ballastDt12_3, ballastDt12_5, ballastDt12_8,
                                           ballastDt13_0, ballastDt13_3, ballastDt13_5, ballastDt13_8))

        ###START OF LADDEN##################################################################################
        ####################################################################################################
        ####################################################################################################
        ####################################################################################################
        ####################################################################################################
        ####################################################################################################
        ####################################################################################################
        ####################################################################################################

        velocitiesL = np.array([k for k in ladenDt if k[5] > 6 and k[5] < 16])[:, 5]
        dataModel = KMeans(n_clusters=4)
        velocitiesL = velocitiesL.reshape(-1, 1)
        dataModel.fit(velocitiesL)
        labels = dataModel.predict(velocitiesL)
        # Extract centroid values

        centroidsL = dataModel.cluster_centers_
        centroidsL = np.sort(centroidsL, axis=0)

        partitionsX = []
        partitionLabels = []
        # For each label
        for curLbl in np.unique(labels):
            # Create a partition for X using records with corresponding label equal to the current
            partitionsX.append(np.asarray(velocitiesL[labels == curLbl]))
            # Create a partition for Y using records with corresponding label equal to the current

            # Keep partition label to ascertain same order of results
            partitionLabels.append(curLbl)

        sorted = []
        initialX = len(partitionsX)
        while sorted.__len__() < initialX:
            min = 100000000000
            for i in range(0, len(partitionsX)):
                mean = np.mean(partitionsX[i])
                if mean < min:
                    min = mean
                    minIndx = i
            sorted.append(partitionsX[minIndx])
            # partitionsX.remove(partitionsX[minIndx])
            partitionsX.pop(minIndx)

        ##delete ladden outliers
        # np.delete(ladenDt, [i for (i, v) in enumerate(ladenDt[:, 8]) if v < (
        # np.mean(ladenDt[:, 8]) - np.std(ladenDt[:, 8])) or v > np.mean(
        # ladenDt[:, 8]) + np.std(
        # ladenDt[:, 8])], 0)

        ###SPEED 10 WIND <1.5
        vel0Min = np.floor(np.min(sorted[0])) + 0.5
        vel1Min = np.floor(np.min(sorted[1])) + 0.5
        vel2Min = np.floor(np.min(sorted[2])) + 0.5
        vel3Min = np.floor(np.min(sorted[3])) + 0.5

        vel0Max = np.floor(np.max(sorted[0])) + 0.5
        vel1Max = np.floor(np.max(sorted[1])) + 0.5
        vel2Max = np.floor(np.max(sorted[2])) + 0.5
        vel3Max = np.floor(np.max(sorted[3])) + 0.5

        vel0Mean = np.floor(np.mean(sorted[0])) + 0.5
        vel1Mean = np.floor(np.mean(sorted[1])) + 0.5
        vel2Mean = np.floor(np.mean(sorted[2])) + 0.5
        vel3Mean = np.floor(np.mean(sorted[3])) + 0.5

        ####VESSEL BASIC INFO
        ldVelocities = []
        ldVelocities.append(vel0Mean)
        ldVelocities.append(vel1Mean)
        ldVelocities.append(vel2Mean)
        ldVelocities.append(vel3Mean)

        # vel0Min = 7
        # vel1Min = 9
        # vel2Min = 12
        # vel3Min = 13.5

        # vel0Max = 12
        # vel1Max = 12.5
        # vel2Max = 13.5
        # vel3Max = 15

        '''workbook._sheets[1]['B6'] = vel0Mean
          workbook._sheets[1]['B16'] = vel1Mean
          workbook._sheets[1]['B26'] = vel2Mean
          workbook._sheets[1]['B36'] = vel3Mean'''

        workbook._sheets[1]['B6'] = str(vel0Mean) + '  (' + str(vel0Min) + ' - ' + str(vel0Max) + ')'
        workbook._sheets[1]['B16'] = str(vel1Mean) + '  (' + str(vel1Min) + ' - ' + str(vel1Max) + ')'
        workbook._sheets[1]['B26'] = str(vel2Mean) + '  (' + str(vel2Min) + ' - ' + str(vel2Max) + ')'
        workbook._sheets[1]['B36'] = str(vel3Mean) + '  (' + str(vel3Min) + ' - ' + str(vel3Max) + ')'

        ##END OF VESSEL BASIC INFO
        speeds = list(itertools.chain(blVelocities, ldVelocities))
        with open('./data/' + company + '/' + vessel + '/ListOfSpeeds.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(
                ['Speeds'])
            for i in range(0, len(speeds)):
                data_writer.writerow(
                    [speeds[i]])

        # workbook._sheets[1]['B6'] = round(centroidsL[0][0], 2)
        # workbook._sheets[1]['B16'] = round(centroidsL[1][0], 2)
        # workbook._sheets[1]['B26'] = round(centroidsL[2][0], 2)
        # workbook._sheets[1]['B36'] = round(centroidsL[3][0], 2)

        ladenDt10_0 = []
        # vel0Min = vel0Min + 2
        centralMean = np.mean(np.array([k for k in ladenDt if
                                        k[5] >= vel0Min and k[5] <= vel0Max and k[8] > 10])[:, 8])
        centralArray = np.array([k for k in ladenDt if
                                 k[5] >= vel0Min and k[5] <= vel0Max and k[8] > 10])[:, 8]
        numberOfApp10_0 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 0 and k[4] <= 1 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[i] and
                                 k[3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 0 and k[4] <= 1 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k >= 3])

            # tlgarrayFoc=np.mean(np.array([k for k in ladenDt if k[5] >= round(centroidsB[0][0], 2) and k[5] <= 10 and k[4] >= 0 and k[4] <= 1 and k[9] > 10])[:, 9])
            tlgarrayFoc = np.array([k for k in ladenDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])

            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array([k for k in ladenDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp10_0.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                numberOfApp10_0.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt10_0.append(round(meanFoc, 2))

        for i in range(9, 14):
            workbook._sheets[1]['B' + str(i)] = ladenDt10_0[i - 9]

        ###SPEED 10  2 < WIND <3
        ladenDt10_3 = []
        numberOfApp10_3 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 2 and k[4] <= 3 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[i] and
                                 k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 2 and k[4] <= 3 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 13])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp10_3.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp10_3.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt10_3.append(round(meanFoc, 2))

        for i in range(9, 14):
            workbook._sheets[1]['C' + str(i)] = ladenDt10_3[i - 9]

        ###SPEED 10  4 < WIND <5
        ladenDt10_5 = []
        numberOfApp10_5 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 4 and k[4] <= 5 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[i] and
                                 k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 4 and k[4] <= 5 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 13])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp10_5.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp10_5.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt10_5.append(round(meanFoc, 2))

        for i in range(9, 14):
            workbook._sheets[1]['D' + str(i)] = ladenDt10_5[i - 9]

        ###SPEED 10  7 < WIND <8
        ladenDt10_8 = []
        numberOfApp10_8 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 6 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[i] and
                                 k[3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 6 and k[5] >= vel0Min and k[5] <= vel0Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])
            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 13])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] >= vel0Min and k[5] <= vel0Max and k[9] >= 3])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp10_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp10_8.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt10_8.append(round(meanFoc, 2))

        for i in range(9, 14):
            workbook._sheets[1]['E' + str(i)] = ladenDt10_8[i - 9]

        ##################################################################################################################
        ##################################################################################################################

        ###SPEED 11.5   WIND <1.5
        centralMean = np.mean(np.array([k for k in ladenDt if
                                        k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 8])
        centralArray = np.array([k for k in ladenDt if
                                 k[5] >= vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 8]
        ladenDt11_0 = []
        numberOfApp11_0 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 0 and k[4] <= 1 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 0 and k[4] <= 1 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp11_0.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp11_0.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt11_0.append(round(meanFoc, 2))

        for i in range(19, 24):
            workbook._sheets[1]['B' + str(i)] = ladenDt11_0[i - 19]

        ###SPEED 11.5  2 < WIND <3
        ladenDt11_3 = []
        numberOfApp11_3 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 2 and k[4] <= 3 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[i] and
                                 k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 2 and k[4] <= 3 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp11_3.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp11_3.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt11_3.append(round(meanFoc, 2))

        for i in range(19, 24):
            workbook._sheets[1]['C' + str(i)] = ladenDt11_3[i - 19]

        ###SPEED 11.5  4 < WIND <5
        ladenDt11_5 = []
        numberOfApp11_5 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 4 and k[4] <= 5 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[i] and
                                 k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 4 and k[4] <= 5 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp11_5.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp11_5.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt11_5.append(round(meanFoc, 2))

        for i in range(19, 24):
            workbook._sheets[1]['D' + str(i)] = ladenDt11_5[i - 19]

            ###SPEED 11.5  7 < WIND <8
        ladenDt11_8 = []
        numberOfApp11_8 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 6 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 6 and k[5] > vel1Min and k[5] <= vel1Max and k[3] >= wind[i] and k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel1Min and k[5] <= vel1Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp11_8.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt11_8.append(round(meanFoc, 2))

        for i in range(19, 24):
            workbook._sheets[1]['E' + str(i)] = ladenDt11_8[i - 19]

        #################################

        ###SPEED 12.5 WIND <1.5
        centralMean = np.mean(np.array([k for k in ladenDt if
                                        k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 8])

        centralArray = np.array([k for k in ladenDt if
                                 k[5] >= vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 8]
        ladenDt12_0 = []
        numberOfApp12_0 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 0 and k[4] <= 1 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[
                                     i] and
                                 k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 0 and k[4] <= 1 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[i] and
                                  k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp12_0.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp12_0.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt12_0.append(round(meanFoc, 2))

        for i in range(29, 34):
            workbook._sheets[1]['B' + str(i)] = ladenDt12_0[i - 29]

        ###SPEED 11.5  2 < WIND <3
        ladenDt12_3 = []
        numberOfApp12_3 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 2 and k[4] <= 3 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 2 and k[4] <= 3 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[i] and
                                  k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp12_3.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp12_3.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt12_3.append(round(meanFoc, 2))

        for i in range(29, 34):
            workbook._sheets[1]['C' + str(i)] = ladenDt12_3[i - 29]

        ###SPEED 11.5  4 < WIND <5
        ladenDt12_5 = []
        numberOfApp12_5 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 4 and k[4] <= 5 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 4 and k[4] <= 5 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[i] and
                                  k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp12_5.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp12_5.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt12_5.append(round(meanFoc, 2))

        for i in range(29, 34):
            workbook._sheets[1]['D' + str(i)] = ladenDt12_5[i - 29]

        ###SPEED 11.5  7 < WIND <8
        ladenDt12_8 = []
        numberOfApp12_8 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 6 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[i] and k[3] <= wind[
                                     i + 1] and k[8] > 10])
            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 6 and k[5] > vel2Min and k[5] <= vel2Max and k[3] >= wind[i] and
                                  k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp12_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp12_8.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt12_8.append(round(meanFoc, 2))

        for i in range(29, 34):
            workbook._sheets[1]['E' + str(i)] = ladenDt12_8[i - 29]

        #################################

        ###SPEED 13.5 WIND <1.5
        centralMean = np.array([k for k in ladenDt if
                                k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])
        if centralMean.__len__() > 0:
            centralMean = np.mean(np.array([k for k in ladenDt if
                                            k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])[:, 8])

            centralArray = np.array([k for k in ladenDt if
                                     k[5] >= vel3Min and k[5] <= vel3Max and k[8] > 4])
        else:
            centralMean = np.mean(np.array([k for k in ladenDt if
                                            k[5] > vel2Min and k[5] <= vel2Max and k[8] > 10])[:, 8]) + 1
            centralArray = np.array([k for k in ladenDt if
                                     k[5] >= vel2Min and k[5] <= vel2Max and k[8] > 4])[:, 8] + 1
        ladenDt13_0 = []
        numberOfApp13_0 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 0 and k[4] <= 1 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >=
                                 wind[
                                     i] and
                                 k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 0 and k[4] <= 1 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[i] and
                                  k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp13_0.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp13_0.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt13_0.append(round(meanFoc, 2))

        for i in range(39, 44):
            workbook._sheets[1]['B' + str(i)] = ladenDt13_0[i - 39]

        ###SPEED 13.5  2 < WIND <3
        ladenDt13_3 = []
        numberOfApp13_3 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 2 and k[4] <= 3 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 2 and k[4] <= 3 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[i] and
                                  k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp13_3.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp13_3.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt13_3.append(round(meanFoc, 2))

        for i in range(39, 44):
            workbook._sheets[1]['C' + str(i)] = ladenDt13_3[i - 39]

        ###SPEED 13.5  4 < WIND <5
        ladenDt13_5 = []
        numberOfApp13_5 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 4 and k[4] <= 5 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])

            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 4 and k[4] <= 5 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[i] and
                                  k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp13_5.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp13_5.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt13_5.append(round(meanFoc, 2))

        for i in range(39, 44):
            workbook._sheets[1]['D' + str(i)] = ladenDt13_5[i - 39]

        ###SPEED 13.5  7 < WIND <8
        ladenDt13_8 = []
        numberOfApp13_8 = []
        for i in range(0, len(wind) - 1):
            arrayFoc = np.array([k for k in ladenDt if
                                 k[4] >= 6 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[
                                     i] and k[
                                     3] <= wind[i + 1] and k[8] > 10])
            steamTime = np.array([k for k in ladenDt if
                                  k[4] >= 6 and k[5] > vel3Min and k[5] <= vel3Max and k[3] >= wind[i] and
                                  k[3] >= wind[
                                      i] and
                                  k[3] <= wind[i + 1] and k[8] > 10])

            tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
            tlgarrayFoc = np.array(
                [k for k in ladenDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])
            # tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
            if tlgarrayFoc.__len__() > lenConditionTlg:
                tlgarrayFoc = np.array(
                    [k for k in ladenDt if k[5] > vel3Min and k[5] <= vel3Max and k[8] > 10])[:, 9]
                meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                    tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (centralMean + np.mean(
                    tlgarrayFoc)) / 2
                numberOfApp13_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
            else:
                weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                  weights=steamTime[:, 12]) if arrayFoc.__len__() > minAccThres else 0
                meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else 0
                numberOfApp13_8.append(arrayFoc.__len__() + centralArray.__len__())
            ladenDt13_8.append(round(meanFoc, 2))

        for i in range(39, 44):
            workbook._sheets[1]['E' + str(i)] = ladenDt13_8[i - 39]

        ####END OF LADDEN #############################################################
        ####END OF LADDEN #############################################################
        ####END OF LADDEN #############################################################
        ##TREAT outliers / missing values for LADDEN values

        # workbook.save(filename=pathToexcel.split('.')[0] + '_1.' + pathToexcel.split('.')[1])
        # return
        #####################################################################################################################
        ##REPLACE NULL ZERO VALUES
        #####################################################################################################################
        ##WEIGHTED MEANS
        values = [k for k in ladenDt10_0 if k != 0]
        length = values.__len__()
        numberOfApp10_0 = [numberOfApp10_0[i] for i in range(0, len(numberOfApp10_0)) if ladenDt10_0[i] > 0]
        for i in range(0, len(ladenDt10_0)):
            if length > 0:
                if ladenDt10_0[i] == 0:
                    ##find items !=0
                    ladenDt10_0[i] = np.array(np.sum(
                        [numberOfApp10_0[i] * values[i] for i in range(0, len(values)) if values[i] > 0])) / np.sum(
                        numberOfApp10_0)
                elif np.isnan(ladenDt10_0[i]):
                    ladenDt10_0[i] = np.array(np.sum(
                        [numberOfApp10_0[i] * values[i] for i in range(0, len(values)) if values[i] > 0])) / np.sum(
                        numberOfApp10_0)

        values = [k for k in ladenDt11_0 if k != 0]
        length = values.__len__()
        numberOfApp11_0 = [numberOfApp11_0[i] for i in range(0, len(numberOfApp11_0)) if ladenDt11_0[i] > 0]
        for i in range(0, len(ladenDt11_0)):
            if length > 0:
                if ladenDt11_0[i] == 0:
                    ##find items !=0
                    ladenDt11_0[i] = np.array(
                        np.sum(
                            [numberOfApp11_0[i] * values[i] for i in range(0, len(values)) if values[i] > 0])) / np.sum(
                        numberOfApp11_0)
                elif np.isnan(ladenDt11_0[i]):
                    ladenDt11_0[i] = np.array(
                        np.sum(
                            [numberOfApp11_0[i] * values[i] for i in range(0, len(values)) if values[i] > 0])) / np.sum(
                        numberOfApp11_0)

        values = [k for k in ladenDt12_0 if k != 0]
        length = values.__len__()
        numberOfApp12_0 = [numberOfApp12_0[i] for i in range(0, len(numberOfApp12_0)) if ladenDt12_0[i] > 0]
        for i in range(0, len(ladenDt12_0)):
            if length > 0:
                if ladenDt12_0[i] == 0:
                    ##find items !=0
                    ladenDt12_0[i] = np.array(
                        np.sum([numberOfApp12_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp12_0)
                elif np.isnan(ladenDt11_0[i]):
                    ladenDt12_0[i] = np.array(
                        np.sum([numberOfApp12_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp12_0)

        values = [k for k in ladenDt13_0 if k != 0]
        length = values.__len__()
        numberOfApp13_0 = [numberOfApp13_0[i] for i in range(0, len(numberOfApp13_0)) if ladenDt13_0[i] > 0]
        for i in range(0, len(ladenDt13_0)):
            if length > 0:
                if ladenDt13_0[i] == 0:
                    ladenDt13_0[i] = np.array(
                        np.sum([numberOfApp13_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp13_0)
                elif np.isnan(ladenDt13_0[i]):
                    ladenDt13_0[i] = np.array(
                        np.sum([numberOfApp13_0[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp13_0)

        values = [k for k in ladenDt10_3 if k != 0]
        length = values.__len__()
        numberOfApp10_3 = [numberOfApp10_3[i] for i in range(0, len(numberOfApp10_3)) if ladenDt10_3[i] > 0]
        for i in range(0, len(ladenDt10_3)):
            if length > 0:
                if ladenDt10_3[i] == 0:
                    ladenDt10_3[i] = np.array(
                        np.sum([numberOfApp10_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp10_3)
                elif np.isnan(ladenDt10_3[i]):
                    ladenDt10_3[i] = np.array(
                        np.sum([numberOfApp10_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp10_3)

        values = [k for k in ladenDt11_3 if k != 0]
        length = values.__len__()
        numberOfApp11_3 = [numberOfApp11_3[i] for i in range(0, len(numberOfApp11_3)) if ladenDt11_3[i] > 0]
        for i in range(0, len(ladenDt11_3)):
            if length > 0:
                if ladenDt11_3[i] == 0:
                    ladenDt11_3[i] = np.array(
                        np.sum([numberOfApp11_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp11_3)
                elif np.isnan(ladenDt11_3[i]):
                    ladenDt11_3[i] = np.array(
                        np.sum([numberOfApp11_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp11_3)

        values = [k for k in ladenDt12_3 if k != 0]
        length = values.__len__()
        numberOfApp12_3 = [numberOfApp12_3[i] for i in range(0, len(numberOfApp12_3)) if ladenDt12_3[i] > 0]
        for i in range(0, len(ladenDt12_3)):
            if length > 0:
                if ladenDt12_3[i] == 0:
                    ladenDt12_3[i] = np.array(
                        np.sum([numberOfApp12_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp12_3)
                elif np.isnan(ladenDt12_3[i]):
                    ladenDt11_3[i] = np.array(
                        np.sum([numberOfApp12_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp12_3)

        values = [k for k in ladenDt13_3 if k != 0]
        length = values.__len__()
        numberOfApp13_3 = [numberOfApp13_3[i] for i in range(0, len(numberOfApp13_3)) if ladenDt13_3[i] > 0]
        for i in range(0, len(ladenDt13_3)):
            if length > 0:
                if ladenDt13_3[i] == 0:
                    ladenDt13_3[i] = np.array(
                        np.sum([numberOfApp13_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp13_3)
                elif np.isnan(ladenDt13_3[i]):
                    ladenDt13_3[i] = np.array(
                        np.sum([numberOfApp13_3[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp13_3)

        values = [k for k in ladenDt10_5 if k != 0]
        length = values.__len__()
        numberOfApp10_5 = [numberOfApp10_5[i] for i in range(0, len(numberOfApp10_5)) if ladenDt10_5[i] > 0]
        for i in range(0, len(ladenDt10_5)):
            if length > 0:
                if ladenDt10_5[i] == 0:
                    ladenDt10_5[i] = np.array(
                        np.sum([numberOfApp10_5[i] * values[i] for i in range(0, len(values)) if
                                ladenDt10_5[i] > 0])) / np.sum(
                        numberOfApp10_5)
                elif np.isnan(ladenDt10_5[i]):
                    ladenDt10_5[i] = np.array(
                        np.sum([numberOfApp10_5[i] * values[i] for i in range(0, len(values)) if
                                ladenDt10_5[i] > 0])) / np.sum(
                        numberOfApp10_5)

        values = [k for k in ladenDt11_5 if k != 0]
        length = values.__len__()
        numberOfApp11_5 = [numberOfApp11_5[i] for i in range(0, len(numberOfApp11_5)) if ladenDt11_5[i] > 0]
        for i in range(0, len(ladenDt11_5)):
            if length > 0:
                if ladenDt11_5[i] == 0:
                    ladenDt11_5[i] = np.array(
                        np.sum([numberOfApp11_5[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp11_5)
                elif np.isnan(ladenDt11_5[i]):
                    ladenDt11_5[i] = np.array(
                        np.sum([numberOfApp11_5[i] * values[i] for i in range(0, len(values))])) / np.sum(
                        numberOfApp11_5)

        values = [k for k in ladenDt12_5 if k != 0]
        length = values.__len__()
        numberOfApp12_5 = [numberOfApp12_5[i] for i in range(0, len(numberOfApp12_5)) if ladenDt12_5[i] > 0]
        for i in range(0, len(ladenDt12_5)):
            if length > 0:
                if ladenDt12_5[i] == 0:
                    ladenDt12_5[i] = np.array(
                        np.sum([numberOfApp12_5[i] * values[i] for i in range(0, len(values)) if
                                ladenDt12_5[i] > 0])) / np.sum(
                        numberOfApp12_5)
                elif np.isnan(ladenDt12_5[i]):
                    ladenDt12_5[i] = np.array(
                        np.sum([numberOfApp12_5[i] * values[i] for i in range(0, len(values)) if
                                ladenDt12_5[i] > 0])) / np.sum(
                        numberOfApp12_5)

        values = [k for k in ladenDt13_5 if k != 0]
        length = values.__len__()
        numberOfApp13_5 = [numberOfApp13_5[i] for i in range(0, len(numberOfApp13_5)) if ladenDt12_5[i] > 0]
        for i in range(0, len(ladenDt13_5)):
            if length > 0:
                if ladenDt13_5[i] == 0:
                    ladenDt13_5[i] = np.array(
                        np.sum([numberOfApp13_5[i] * values[i] for i in range(0, len(values)) if
                                ladenDt13_5[i] > 0])) / np.sum(
                        numberOfApp13_5)
                elif np.isnan(ladenDt13_5[i]):
                    ladenDt13_5[i] = np.array(
                        np.sum([numberOfApp13_5[i] * values[i] for i in range(0, len(values)) if
                                ladenDt13_5[i] > 0])) / np.sum(
                        numberOfApp13_5)

        values = [k for k in ladenDt10_8 if k != 0]
        length = values.__len__()
        numberOfApp10_8 = [numberOfApp10_8[i] for i in range(0, len(numberOfApp10_8)) if ladenDt10_8[i] > 0]
        for i in range(0, len(ladenDt10_8)):
            if length > 0:
                if ladenDt10_8[i] == 0:
                    ladenDt10_8[i] = np.array(
                        np.sum([numberOfApp10_8[i] * values[i] for i in range(0, len(values)) if
                                ladenDt10_8[i] > 0])) / np.sum(
                        numberOfApp10_8)
                elif np.isnan(ladenDt10_8[i]):
                    ladenDt10_8[i] = np.array(
                        np.sum([numberOfApp10_8[i] * values[i] for i in range(0, len(values)) if
                                ladenDt10_8[i] > 0])) / np.sum(
                        numberOfApp10_8)

        values = [k for k in ladenDt11_8 if k != 0]
        length = values.__len__()
        numberOfApp11_8 = [numberOfApp11_8[i] for i in range(0, len(numberOfApp11_8)) if ladenDt11_8[i] > 0]
        for i in range(0, len(ladenDt11_8)):
            if length > 0:
                if length > 0:
                    if ladenDt11_8[i] == 0:
                        ladenDt11_8[i] = np.array(
                            np.sum([numberOfApp11_8[i] * values[i] for i in range(0, len(values)) if
                                    ladenDt11_8[i] > 0])) / np.sum(
                            numberOfApp11_8)
                    elif np.isnan(ladenDt11_8[i]):
                        ladenDt11_8[i] = np.array(
                            np.sum([numberOfApp11_8[i] * values[i] for i in range(0, len(values)) if
                                    ladenDt11_8[i] > 0])) / np.sum(
                            numberOfApp11_8)

        values = [k for k in ladenDt12_8 if k != 0]
        length = values.__len__()
        numberOfApp12_8 = [numberOfApp12_8[i] for i in range(0, len(numberOfApp12_8)) if ladenDt12_8[i] > 0]
        for i in range(0, len(ladenDt12_8)):
            if length > 0:
                if ladenDt12_8[i] == 0:
                    ladenDt12_8[i] = np.array(
                        np.sum([numberOfApp12_8[i] * values[i] for i in range(0, len(values)) if
                                ladenDt12_8[i] > 0])) / np.sum(
                        numberOfApp12_8)
                elif np.isnan(ladenDt12_8[i]):
                    ladenDt12_8[i] = np.array(
                        np.sum([numberOfApp12_8[i] * values[i] for i in range(0, len(values)) if
                                ladenDt12_8[i] > 0])) / np.sum(
                        numberOfApp12_8)

        values = [k for k in ladenDt13_8 if k != 0]
        length = values.__len__()
        numberOfApp13_8 = [numberOfApp13_8[i] for i in range(0, len(numberOfApp13_8)) if ladenDt13_8[i] > 0]
        for i in range(0, len(ladenDt13_8)):
            if length > 0:
                if ladenDt13_8[i] == 0:
                    ladenDt13_8[i] = np.array(
                        np.sum([numberOfApp13_8[i] * values[i] for i in range(0, len(values)) if
                                ladenDt13_8[i] > 0])) / np.sum(
                        numberOfApp13_8)
                elif np.isnan(ladenDt13_8[i]):
                    ladenDt13_8[i] = np.array(
                        np.sum([numberOfApp12_8[i] * values[i] for i in range(0, len(values)) if
                                ladenDt13_8[i] > 0])) / np.sum(
                        numberOfApp13_8)

        #####################################################################################################################
        if (np.array(ladenDt10_0) == 0).all() and (np.array(ladenDt10_3) == 0).all() and (
                np.array(ladenDt10_5) == 0).all() and (np.array(ladenDt10_8) == 0).all():
            ladenDt10_0 = np.array(ladenDt11_0) - 2
            ladenDt10_3 = np.array(ladenDt11_3) - 2
            ladenDt10_5 = np.array(ladenDt11_5) - 2
            ladenDt10_8 = np.array(ladenDt11_8) - 2

        if (np.array(ladenDt10_3) == 0).all():
            ladenDt10_3 = np.array(ladenDt10_0) + 1.5

        if (np.array(ladenDt10_5) == 0).all():
            ladenDt10_5 = np.array(ladenDt10_3) + 1.5

        if (np.array(ladenDt10_8) == 0).all():
            ladenDt10_8 = np.array(ladenDt10_5) + 1.5

        if (np.array(ladenDt11_0) == 0).all():
            ladenDt11_0 = np.array(ladenDt10_0) + 1

        if (np.array(ladenDt11_3) == 0).all():
            ladenDt11_3 = np.array(ladenDt11_0) + 1.5

        if (np.array(ladenDt11_5) == 0).all():
            ladenDt11_5 = np.array(ladenDt11_3) + 1.5

        if (np.array(ladenDt11_8) == 0).all():
            ladenDt11_8 = np.array(ladenDt11_5) + 1.5

        if (np.array(ladenDt12_0) == 0).all():
            ladenDt12_0 = np.array(ladenDt11_0) + 1

        if (np.array(ladenDt12_3) == 0).all():
            ladenDt12_3 = np.array(ladenDt12_0) + 1.5
        for i in range(0, len(ladenDt12_3)):
            if ladenDt12_3[i] == inf:
                ladenDt12_3[i] = 0
        if (np.array(ladenDt12_5) == 0).all():
            ladenDt12_5 = np.array(ladenDt12_3) + 1.5

        if (np.array(ladenDt12_8) == 0).all():
            ladenDt12_8 = np.array(ladenDt12_5) + 1.5

        if (np.array(ladenDt13_0) == 0).all():
            ladenDt13_0 = np.array(ladenDt12_0) + 1

        if (np.array(ladenDt13_3) == 0).all():
            ladenDt13_3 = np.array(ladenDt13_0) + 1.5

        if (np.array(ladenDt13_5) == 0).all():
            ladenDt13_5 = np.array(ladenDt13_3) + 1.5

        if (np.array(ladenDt13_8) == 0).all():
            ladenDt13_8 = np.array(ladenDt13_5) + 1.5
        ######################
        ####################

        values = [k for k in ladenDt10_0 if k != 0]
        if values.__len__() == 0:
            if (np.array(ladenDt10_3) != 0).all():
                ladenDt10_0 = np.array(ladenDt10_3) - 1
            elif (np.array(ladenDt10_5) != 0).all():
                ladenDt10_0 = np.array(ladenDt10_5) - 2
            else:
                ladenDt10_0 = np.array(ladenDt10_8) - 3

        values = [k for k in ladenDt11_0 if k != 0]
        if values.__len__() == 0:
            ladenDt11_0 = np.array(ladenDt11_3) - 1
        values = [k for k in ladenDt12_0 if k != 0]

        if values.__len__() == 0:
            ladenDt12_0 = np.array(ladenDt12_3) - 1

        values = [k for k in ladenDt13_0 if k != 0]
        if values.__len__() == 0:
            ladenDt13_0 = np.array(ladenDt13_3) - 1

        for i in range(0, len(ladenDt10_0)):

            if (ladenDt10_0[i] >= ladenDt11_0[i]) and ladenDt11_0[i] > 0:
                while (ladenDt10_0[i] >= ladenDt11_0[i]):
                    ladenDt11_0[i] = ladenDt11_0[i] + 0.1 * ladenDt11_0[i]

        for i in range(0, len(ladenDt11_0)):

            if (ladenDt11_0[i] >= ladenDt12_0[i]) and ladenDt12_0[i] > 0:
                while (ladenDt11_0[i] >= ladenDt12_0[i]):
                    ladenDt12_0[i] = ladenDt12_0[i] + 0.1 * ladenDt12_0[i]

        for i in range(0, len(ladenDt12_0)):

            if (ladenDt12_0[i] >= ladenDt13_0[i]) and ladenDt13_0[i] > 0:
                while (ladenDt12_0[i] >= ladenDt13_0[i]):
                    ladenDt13_0[i] = ladenDt13_0[i] + 0.1 * ladenDt13_0[i]

        #####################################################################################################################

        for i in range(0, len(ladenDt10_3)):

            if (ladenDt10_0[i] >= ladenDt10_3[i]):
                while (ladenDt10_0[i] >= ladenDt10_3[i]):
                    if ladenDt10_3[i] == 0:
                        ladenDt10_3[i] = ladenDt10_0[i] + 0.1 * ladenDt10_0[i]
                    else:
                        ladenDt10_3[i] = ladenDt10_3[i] + 0.1 * ladenDt10_3[i]

        for i in range(0, len(ladenDt10_5)):

            if (ladenDt10_3[i] >= ladenDt10_5[i]):
                while (ladenDt10_3[i] >= ladenDt10_5[i]):
                    if ladenDt10_5[i] == 0:
                        ladenDt10_5[i] = ladenDt10_3[i] + 0.1 * ladenDt10_3[i]
                    else:
                        ladenDt10_5[i] = ladenDt10_5[i] + 0.1 * ladenDt10_5[i]

        for i in range(0, len(ladenDt10_8)):

            if (ladenDt10_5[i] >= ladenDt10_8[i]):
                while (ladenDt10_5[i] >= ladenDt10_8[i]):
                    if ladenDt10_8[i] == 0:
                        ladenDt10_8[i] = ladenDt10_5[i] + 0.1 * ladenDt10_5[i]
                    else:
                        ladenDt10_8[i] = ladenDt10_8[i] + 0.1 * ladenDt10_8[i]
        #####################################################################################################################

        for i in range(0, len(ladenDt11_3)):

            if (ladenDt11_0[i] >= ladenDt11_3[i]):
                while (ladenDt11_0[i] >= ladenDt11_3[i]):
                    if ladenDt11_3[i] == 0:
                        ladenDt11_3[i] = ladenDt11_0[i] + 0.1 * ladenDt11_0[i]
                    else:
                        ladenDt11_3[i] = ladenDt11_3[i] + 0.1 * ladenDt11_3[i]

        for i in range(0, len(ladenDt11_5)):

            if (ladenDt11_3[i] >= ladenDt11_5[i]):
                while (ladenDt11_3[i] >= ladenDt11_5[i]):
                    if ladenDt11_5[i] == 0:
                        ladenDt11_5[i] = ladenDt11_3[i] + 0.1 * ladenDt11_3[i]
                    else:
                        ladenDt11_5[i] = ladenDt11_5[i] + 0.1 * ladenDt11_5[i]

        for i in range(0, len(ladenDt11_8)):

            if (ladenDt11_5[i] >= ladenDt11_8[i]):
                while (ladenDt11_5[i] >= ladenDt11_8[i]):
                    if ladenDt11_8[i] == 0:
                        ladenDt11_8[i] = ladenDt11_5[i] + 0.1 * ladenDt11_5[i]
                    else:
                        ladenDt11_8[i] = ladenDt11_8[i] + 0.1 * ladenDt11_8[i]

        #####################################################################################################################

        for i in range(0, len(ladenDt10_3)):

            if (ladenDt10_3[i] > ladenDt11_3[i]) and ladenDt11_3[i] > 0:
                while (ladenDt10_3[i] > ladenDt11_3[i]):
                    ladenDt11_3[i] = ladenDt11_3[i] + 0.1 * ladenDt11_3[i]

        for i in range(0, len(ladenDt10_5)):

            if (ladenDt10_5[i] > ladenDt11_5[i]) and ladenDt11_5[i] > 0:
                while (ladenDt10_5[i] > ladenDt11_5[i]):
                    ladenDt11_5[i] = ladenDt11_5[i] + 0.1 * ladenDt11_5[i]

        for i in range(0, len(ladenDt10_5)):

            if (ladenDt10_8[i] > ladenDt11_8[i]) and ladenDt11_8[i] > 0:
                while (ladenDt10_8[i] > ladenDt11_8[i]):
                    ladenDt11_8[i] = ladenDt11_8[i] + 0.1 * ladenDt11_8[i]
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #####################################################################################################################

        for i in range(0, len(ladenDt12_3)):

            if (ladenDt12_0[i] > ladenDt12_3[i]):
                while (ladenDt12_0[i] > ladenDt12_3[i]):
                    if ladenDt12_3[i] == 0:
                        ladenDt12_3[i] = ladenDt12_0[i] + 0.1 * ladenDt12_0[i]
                    else:
                        ladenDt12_3[i] = ladenDt12_3[i] + 0.1 * ladenDt12_3[i]

        for i in range(0, len(ladenDt12_5)):

            if (ladenDt12_3[i] > ladenDt12_5[i]):
                while (ladenDt12_3[i] > ladenDt12_5[i]):
                    if ladenDt12_5[i] == 0:
                        ladenDt12_5[i] = ladenDt12_3[i] + 0.1 * ladenDt12_3[i]
                    else:
                        ladenDt12_5[i] = ladenDt12_5[i] + 0.1 * ladenDt12_5[i]

        for i in range(0, len(ladenDt12_8)):

            if (ladenDt12_5[i] > ladenDt12_8[i]):
                while (ladenDt12_5[i] > ladenDt12_8[i]):
                    if ladenDt12_8[i] == 0:
                        ladenDt12_8[i] = ladenDt12_5[i] + 0.1 * ladenDt12_5[i]
                    else:
                        ladenDt12_8[i] = ladenDt12_8[i] + 0.1 * ladenDt12_8[i]

        #####################################################################################################################
        for i in range(0, len(ladenDt12_3)):

            if (ladenDt11_3[i] > ladenDt12_3[i]) and ladenDt12_3[i] > 0:
                while (ladenDt11_3[i] > ladenDt12_3[i]):
                    ladenDt12_3[i] = ladenDt12_3[i] + 0.1 * ladenDt12_3[i]

        for i in range(0, len(ladenDt12_5)):

            if (ladenDt11_5[i] > ladenDt12_5[i]) and ladenDt12_5[i] > 0:
                while (ladenDt11_5[i] > ladenDt12_5[i]):
                    ladenDt12_5[i] = ladenDt12_5[i] + 0.1 * ladenDt12_5[i]

        for i in range(0, len(ladenDt12_8)):

            if (ladenDt11_8[i] > ladenDt12_8[i]) and ladenDt12_8[i] > 0:
                while (ladenDt11_8[i] > ladenDt12_8[i]):
                    ladenDt12_8[i] = ladenDt12_8[i] + 0.1 * ladenDt12_8[i]
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # STANDARIZE WEATHER
        #####################################################################################################################

        for i in range(0, len(ladenDt13_3)):

            if (ladenDt13_0[i] > ladenDt13_3[i]):
                while (ladenDt13_0[i] > ladenDt13_3[i]):
                    if ladenDt13_3[i] == 0:
                        ladenDt13_3[i] = ladenDt12_0[i] + 0.1 * ladenDt12_0[i]
                    else:
                        ladenDt13_3[i] = ladenDt13_3[i] + 0.1 * ladenDt13_3[i]

        for i in range(0, len(ladenDt13_5)):

            if (ladenDt13_3[i] > ladenDt13_5[i]):
                while (ladenDt13_3[i] > ladenDt13_5[i]):
                    if ladenDt13_5[i] == 0:
                        ladenDt13_5[i] = ladenDt12_3[i] + 0.1 * ladenDt12_3[i]
                    else:
                        ladenDt13_5[i] = ladenDt13_5[i] + 0.1 * ladenDt13_5[i]

        for i in range(0, len(ladenDt13_8)):

            if (ladenDt13_5[i] > ladenDt13_8[i]):
                while (ladenDt13_5[i] > ladenDt13_8[i]):
                    if ladenDt13_8[i] == 0:
                        ladenDt13_8[i] = ladenDt12_5[i] + 0.1 * ladenDt12_5[i]
                    else:
                        ladenDt13_8[i] = ladenDt13_8[i] + 0.1 * ladenDt13_8[i]

        #####################################################################################################################

        for i in range(0, len(ladenDt13_3)):

            if (ladenDt12_3[i] > ladenDt13_3[i]) and ladenDt13_3[i] > 0:
                while (ladenDt12_3[i] > ladenDt13_3[i]):
                    ladenDt13_3[i] = ladenDt13_3[i] + 0.1 * ladenDt13_3[i]

        for i in range(0, len(ladenDt13_5)):

            if (ladenDt12_5[i] > ladenDt13_5[i]) and ladenDt13_5[i] > 0:
                while (ladenDt12_5[i] > ladenDt13_5[i]):
                    ladenDt13_5[i] = ladenDt13_5[i] + 0.1 * ladenDt13_5[i]

        for i in range(0, len(ladenDt13_8)):

            if (ladenDt12_8[i] > ladenDt13_8[i]) and ladenDt13_8[i] > 0:
                while (ladenDt12_8[i] > ladenDt13_8[i]):
                    ladenDt13_8[i] = ladenDt13_8[i] + 0.1 * ladenDt13_8[i]
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ##LAST CHECKPOINT
        for i in range(0, len(ladenDt13_8)):

            if (ladenDt13_5[i] > ladenDt13_8[i]):
                while (ladenDt13_5[i] > ladenDt13_8[i]):
                    if ladenDt13_8[i] == 0:
                        ladenDt13_8[i] = ladenDt12_5[i] + 0.1 * ladenDt12_5[i]
                    else:
                        ladenDt13_8[i] = ladenDt13_8[i] + 0.1 * ladenDt13_8[i]
        # 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ### fix side /against LADEN
        if (ladenDt10_0[0] <= ladenDt10_0[2]):
            while (ladenDt10_0[0] <= ladenDt10_0[2]):
                ladenDt10_0[0] = ladenDt10_0[0] + 0.1 * ladenDt10_0[0]

        if (ladenDt10_0[1] <= ladenDt10_0[3]):
            while (ladenDt10_0[1] <= ladenDt10_0[3]):
                ladenDt10_0[1] = ladenDt10_0[1] + 0.1 * ladenDt10_0[1]

        if (ladenDt10_3[0] <= ladenDt10_3[2]):
            while (ladenDt10_3[0] <= ladenDt10_3[2]):
                ladenDt10_3[0] = ladenDt10_3[0] + 0.1 * ladenDt10_3[0]

        if (ladenDt10_3[1] <= ladenDt10_3[3]):
            while (ladenDt10_3[1] <= ladenDt10_3[3]):
                ladenDt10_3[1] = ladenDt10_3[1] + 0.1 * ladenDt10_3[1]

        if (ladenDt10_5[0] <= ladenDt10_5[2]):
            while (ladenDt10_5[0] <= ladenDt10_5[2]):
                ladenDt10_5[0] = ladenDt10_5[0] + 0.1 * ladenDt10_5[0]

        if (ladenDt10_5[1] <= ladenDt10_5[3]):
            while (ladenDt10_5[1] <= ladenDt10_5[3]):
                ladenDt10_5[1] = ladenDt10_5[1] + 0.1 * ladenDt10_5[1]

        if (ladenDt10_8[0] <= ladenDt10_8[2]):
            while (ladenDt10_8[0] <= ladenDt10_8[2]):
                ladenDt10_8[0] = ladenDt10_8[0] + 0.1 * ladenDt10_8[0]

        if (ladenDt10_8[1] <= ladenDt10_8[3]):
            while (ladenDt10_8[1] <= ladenDt10_8[3]):
                ladenDt10_8[1] = ladenDt10_8[1] + 0.1 * ladenDt10_8[1]

        if (ladenDt11_0[0] <= ladenDt11_0[2]):
            while (ladenDt11_0[0] <= ladenDt11_0[2]):
                ladenDt11_0[0] = ladenDt11_0[0] + 0.1 * ladenDt11_0[0]

        if (ladenDt11_0[1] <= ladenDt11_0[3]):
            while (ladenDt11_0[1] <= ladenDt11_0[3]):
                ladenDt11_0[1] = ladenDt11_0[1] + 0.1 * ladenDt11_0[1]

        if (ladenDt11_3[0] <= ladenDt11_3[2]):
            while (ladenDt11_3[0] <= ladenDt11_3[2]):
                ladenDt11_3[0] = ladenDt11_3[0] + 0.1 * ladenDt11_3[0]

        if (ladenDt11_3[1] <= ladenDt11_3[3]):
            while (ladenDt11_3[1] <= ladenDt11_3[3]):
                ladenDt11_3[1] = ladenDt11_3[1] + 0.1 * ladenDt11_3[1]

        if (ladenDt11_5[0] <= ladenDt11_5[2]):
            while (ladenDt11_5[0] <= ladenDt11_5[2]):
                ladenDt11_5[0] = ladenDt11_5[0] + 0.1 * ladenDt11_5[0]

        if (ladenDt11_5[1] <= ladenDt11_5[3]):
            while (ladenDt11_5[1] <= ladenDt11_5[3]):
                ladenDt11_5[1] = ladenDt11_5[1] + 0.1 * ladenDt11_5[1]

        if (ladenDt11_8[0] <= ladenDt11_8[2]):
            while (ladenDt11_8[0] <= ladenDt11_8[2]):
                ladenDt11_8[0] = ladenDt11_8[0] + 0.1 * ladenDt11_8[0]

        if (ladenDt11_8[1] <= ladenDt11_8[3]):
            while (ladenDt11_8[1] <= ladenDt11_8[3]):
                ladenDt11_8[1] = ladenDt11_8[1] + 0.1 * ladenDt11_8[1]

        if (ladenDt12_0[0] <= ladenDt12_0[2]):
            while (ladenDt12_0[0] <= ladenDt12_0[2]):
                ladenDt12_0[0] = ladenDt12_0[0] + 0.1 * ladenDt12_0[0]

        if (ladenDt12_0[1] <= ladenDt12_0[3]):
            while (ladenDt12_0[1] <= ladenDt12_0[3]):
                ladenDt12_0[1] = ladenDt12_0[1] + 0.1 * ladenDt12_0[1]

        if (ladenDt12_3[0] <= ladenDt12_3[2]):
            while (ladenDt12_3[0] <= ladenDt12_3[2]):
                ladenDt12_3[0] = ladenDt12_3[0] + 0.1 * ladenDt12_3[0]

        if (ladenDt12_3[1] <= ladenDt12_3[3]):
            while (ladenDt12_3[1] <= ladenDt12_3[3]):
                ladenDt12_3[1] = ladenDt12_3[1] + 0.1 * ladenDt12_3[1]

        if (ladenDt12_5[0] <= ladenDt12_5[2]):
            while (ladenDt12_5[0] <= ladenDt12_5[2]):
                ladenDt12_5[0] = ladenDt12_5[0] + 0.1 * ladenDt12_5[0]

        if (ladenDt12_5[1] <= ladenDt12_5[3]):
            while (ladenDt12_5[1] <= ladenDt12_5[3]):
                ladenDt12_5[1] = ladenDt12_5[1] + 0.1 * ladenDt12_5[1]

        if (ladenDt12_8[0] <= ladenDt12_8[2]):
            while (ladenDt12_8[0] <= ladenDt12_8[2]):
                ladenDt12_8[0] = ladenDt12_8[0] + 0.1 * ladenDt12_8[0]

        if (ladenDt12_8[1] <= ladenDt12_8[3]):
            while (ladenDt12_8[1] <= ladenDt12_8[3]):
                ladenDt12_8[1] = ladenDt12_8[1] + 0.1 * ladenDt12_8[1]

        if (ladenDt13_0[0] <= ladenDt13_0[2]):
            while (ladenDt13_0[0] <= ladenDt13_0[2]):
                ladenDt13_0[0] = ladenDt13_0[0] + 0.1 * ladenDt13_0[0]

        if (ladenDt13_0[1] <= ladenDt13_0[3]):
            while (ladenDt13_0[1] <= ladenDt13_0[3]):
                ladenDt13_0[1] = ladenDt13_0[1] + 0.1 * ladenDt13_0[1]

        if (ladenDt13_3[0] <= ladenDt13_3[2]):
            while (ladenDt13_3[0] <= ladenDt13_3[2]):
                ladenDt13_3[0] = ladenDt13_3[0] + 0.1 * ladenDt13_3[0]

        if (ladenDt13_3[1] <= ladenDt13_3[3]):
            while (ladenDt13_3[1] <= ladenDt13_3[3]):
                ladenDt13_3[1] = ladenDt13_3[1] + 0.1 * ladenDt13_3[1]

        if (ladenDt13_5[0] <= ladenDt13_5[2]):
            while (ladenDt13_5[0] <= ladenDt13_5[2]):
                ladenDt13_5[0] = ladenDt13_5[0] + 0.1 * ladenDt13_5[0]

        if (ladenDt13_5[1] <= ladenDt13_5[3]):
            while (ladenDt13_5[1] <= ladenDt13_5[3]):
                ladenDt13_5[1] = ladenDt13_5[1] + 0.1 * ladenDt13_5[1]

        if (ladenDt13_8[0] <= ladenDt13_8[2]):
            while (ladenDt13_8[0] <= ladenDt13_8[2]):
                ladenDt13_8[0] = ladenDt13_8[0] + 0.1 * ladenDt13_8[0]

        if (ladenDt13_8[1] <= ladenDt13_8[3]):
            while (ladenDt13_8[1] <= ladenDt13_8[3]):
                ladenDt13_8[1] = ladenDt13_8[1] + 0.1 * ladenDt13_8[1]
        ##################################################END AGAINST SIDE#################################
        if (np.array(ladenDt10_0) == 0).all():
            ladenDt10_0 = np.array(ladenDt10_3) - 1 if (np.array(ladenDt10_0) != 3).all() else np.array(ladenDt10_5) - 2
        for i in range(9, 14):
            ladenDt10_0[0] = ladenDt10_0[4] + 1 if ladenDt10_0[0] <= ladenDt10_0[4] else ladenDt10_0[0]
            ladenDt10_0[2] = ladenDt10_0[3] + 1 if ladenDt10_0[2] <= ladenDt10_0[3] else ladenDt10_0[2]
            workbook._sheets[1]['B' + str(i)] = round(ladenDt10_0[i - 9], 2)

            ##TREAT outliers / missing values for ladent values
        if (np.array(ladenDt10_3) == 0).all():
            ladenDt10_3 = np.array(ladenDt10_5) - 1 if (np.array(ladenDt10_5) != 0).any() else np.array(ladenDt10_8) - 2
        for i in range(9, 14):
            ladenDt10_3[0] = ladenDt10_3[4] + 1 if ladenDt10_3[0] <= ladenDt10_3[4] else ladenDt10_3[0]
            ladenDt10_3[2] = ladenDt10_3[3] + 1 if ladenDt10_3[2] <= ladenDt10_3[3] else ladenDt10_3[2]
            workbook._sheets[1]['C' + str(i)] = round(ladenDt10_3[i - 9], 2)

            ##TREAT outliers / missing values for ladent values
        if (np.array(ladenDt10_5) == 0).all():
            ladenDt10_5 = np.array(ladenDt10_8) - 1 if (np.array(ladenDt10_8) != 0).any() else np.array(ladenDt10_3) + 1
        for i in range(9, 14):
            ladenDt10_5[0] = ladenDt10_5[4] + 1 if ladenDt10_5[0] <= ladenDt10_5[4] else ladenDt10_5[0]
            ladenDt10_5[2] = ladenDt10_5[3] + 1 if ladenDt10_5[2] <= ladenDt10_5[3] else ladenDt10_5[2]
            workbook._sheets[1]['D' + str(i)] = round(ladenDt10_5[i - 9], 2)

            ##TREAT outliers / missing values for ladent values
        if (np.array(ladenDt10_8) == 0).all():
            ladenDt10_8 = np.array(ladenDt10_5) + 1 if (np.array(ladenDt10_5) != 0).any() else np.array(ladenDt10_3) + 3
        for i in range(9, 14):
            ladenDt10_8[0] = ladenDt10_8[4] + 1 if ladenDt10_8[0] <= ladenDt10_8[4] else ladenDt10_8[0]
            ladenDt10_8[2] = ladenDt10_8[3] + 1 if ladenDt10_8[2] <= ladenDt10_8[3] else ladenDt10_8[2]
            workbook._sheets[1]['E' + str(i)] = round(ladenDt10_8[i - 9], 2)

            ####################################################################################################

            # values = [k for k in ladenDt11_0 if k != 0]
            # length = values.__len__()

        for i in range(19, 24):
            ladenDt11_0[0] = ladenDt11_0[4] + 1 if ladenDt11_0[0] <= ladenDt11_0[4] else ladenDt11_0[0]
            ladenDt11_0[2] = ladenDt11_0[3] + 1 if ladenDt11_0[2] <= ladenDt11_0[3] else ladenDt11_0[2]
            workbook._sheets[1]['B' + str(i)] = round(ladenDt11_0[i - 19], 2)

            ##TREAT outliers / missing values for ladent values

        for i in range(19, 24):
            ladenDt11_8[0] = ladenDt11_8[4] + 1 if ladenDt11_8[0] <= ladenDt11_8[4] else ladenDt11_8[0]
            ladenDt11_8[2] = ladenDt11_8[3] + 1 if ladenDt11_8[2] <= ladenDt11_8[3] else ladenDt11_8[2]
            workbook._sheets[1]['C' + str(i)] = round(ladenDt11_3[i - 19], 2)

            ##TREAT outliers / missing values for ladent values

        for i in range(19, 24):
            ladenDt11_5[0] = ladenDt11_5[4] + 1 if ladenDt11_5[0] <= ladenDt11_5[4] else ladenDt11_5[0]
            ladenDt11_5[2] = ladenDt11_5[3] + 1 if ladenDt11_5[2] <= ladenDt11_5[3] else ladenDt11_5[2]
            workbook._sheets[1]['D' + str(i)] = round(ladenDt11_5[i - 19], 2)

            ##TREAT outliers / missing values for ladent values

        for i in range(19, 24):
            ladenDt11_8[0] = ladenDt11_8[4] + 1 if ladenDt11_8[0] <= ladenDt11_8[4] else ladenDt11_8[0]
            ladenDt11_8[2] = ladenDt11_8[3] + 1 if ladenDt11_8[2] <= ladenDt11_8[3] else ladenDt11_8[2]
            workbook._sheets[1]['E' + str(i)] = round(ladenDt11_8[i - 19], 2)

            ####################################################################################################

        for i in range(29, 34):
            ladenDt12_0[0] = ladenDt12_0[4] + 1 if ladenDt12_0[0] <= ladenDt12_0[4] else ladenDt12_0[0]
            ladenDt12_0[2] = ladenDt12_0[3] + 1 if ladenDt12_0[2] <= ladenDt12_0[3] else ladenDt12_0[2]
            workbook._sheets[1]['B' + str(i)] = round(ladenDt12_0[i - 29], 2)

            ##TREAT outliers / missing values for ladent values

        for i in range(29, 34):
            ladenDt12_3[0] = ladenDt12_3[4] + 1 if ladenDt12_3[0] <= ladenDt12_3[4] else ladenDt12_3[0]
            ladenDt12_3[2] = ladenDt12_3[3] + 1 if ladenDt12_3[2] <= ladenDt12_3[3] else ladenDt12_3[2]
            workbook._sheets[1]['C' + str(i)] = round(ladenDt12_3[i - 29], 2)

            ##TREAT outliers / missing values for ladent values

        for i in range(29, 34):
            ladenDt12_5[0] = ladenDt12_5[4] + 1 if ladenDt12_5[0] <= ladenDt12_5[4] else ladenDt12_5[0]
            ladenDt12_5[2] = ladenDt12_5[3] + 1 if ladenDt12_5[2] <= ladenDt12_5[3] else ladenDt12_5[2]
            workbook._sheets[1]['D' + str(i)] = round(ladenDt12_5[i - 29], 2)

            ##TREAT outliers / missing values for ladent values

        for i in range(29, 34):
            ladenDt12_8[0] = ladenDt12_8[4] + 1 if ladenDt12_8[0] <= ladenDt12_8[4] else ladenDt12_8[0]
            ladenDt12_8[2] = ladenDt12_8[3] + 1 if ladenDt12_8[2] <= ladenDt12_8[3] else ladenDt12_8[2]
            workbook._sheets[1]['E' + str(i)] = round(ladenDt12_8[i - 29], 2)

            ################################################################################################################
        for i in range(39, 44):
            ladenDt13_0[0] = ladenDt13_0[4] + 1 if ladenDt13_0[0] <= ladenDt13_0[4] else ladenDt13_0[0]
            ladenDt13_0[2] = ladenDt13_0[3] + 1 if ladenDt13_0[2] <= ladenDt13_0[3] else ladenDt13_0[2]
            workbook._sheets[1]['B' + str(i)] = round(ladenDt13_0[i - 39], 2)

            ##TREAT outliers / missing values for ladent values

        for i in range(39, 44):
            ladenDt13_3[0] = ladenDt13_3[4] + 1 if ladenDt13_3[0] <= ladenDt13_3[4] else ladenDt13_3[0]
            ladenDt13_3[2] = ladenDt13_3[3] + 1 if ladenDt13_3[2] <= ladenDt13_3[3] else ladenDt13_3[2]
            workbook._sheets[1]['C' + str(i)] = round(ladenDt13_3[i - 39], 2)

            ##TREAT outliers / missing values for ladent values

        for i in range(39, 44):
            ladenDt13_5[0] = ladenDt13_5[4] + 1 if ladenDt13_5[0] <= ladenDt13_5[4] else ladenDt13_5[0]
            ladenDt13_5[2] = ladenDt13_5[3] + 1 if ladenDt13_5[2] <= ladenDt13_5[3] else ladenDt13_5[2]
            workbook._sheets[1]['D' + str(i)] = round(ladenDt13_5[i - 39], 2)

            ##TREAT outliers / missing values for ladent values

        for i in range(39, 44):
            ladenDt13_8[0] = ladenDt13_8[4] + 1 if ladenDt13_8[0] <= ladenDt13_8[4] else ladenDt13_8[0]
            ladenDt13_8[2] = ladenDt13_8[3] + 1 if ladenDt13_8[2] <= ladenDt13_8[3] else ladenDt13_8[2]
            workbook._sheets[1]['E' + str(i)] = round(ladenDt13_8[i - 39], 2)

        ladenCons = list(itertools.chain(ladenDt10_0, ladenDt10_3, ladenDt10_5, ladenDt10_8,
                                         ladenDt11_0, ladenDt11_3, ladenDt11_5, ladenDt10_8,
                                         ladenDt12_0, ladenDt12_3, ladenDt12_5, ladenDt12_8,
                                         ladenDt13_0, ladenDt13_3, ladenDt13_5, ladenDt13_8))

        profCons = list(itertools.chain(ballastCons, ladenCons))

        with open('./data/' + company + '/' + vessel + '/ListOfCons.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(
                ['FOC'])
            for i in range(0, len(profCons)):
                data_writer.writerow(
                    [profCons[i]])

        workbook.save(filename=pathToexcel.split('.')[0] + '_1.' + pathToexcel.split('.')[1])
        return

    def fillDetailedExcelProfCons(self, company, vessel, pathToexcel, dataSet, rawData, tlgDataset, dataSetBDD,
                                  dataSetADD):
        ##FEATURE SET EXACT POSITION OF COLUMNS NEEDED IN ORDER TO PRODUCE EXCEL
        # 2nd place BALLAST FLAG
        # 8th place DRAFT
        # 10th place WD
        # 11th place WF
        # 12th place SPEED
        # 15th place ME FOC 24H
        # 16th place ME FOC 24H TLGS
        # 17th place TRIM
        # 19th place SteamHours
        # 18th place STW_TLG
        # 20st place swellSWH
        # dataSet = dataSet[dataSet[:, 22] > 0]
        '''foc = np.array([k for k in dataSet if float(k[15]) > 0])[:, 15]
          df = pd.DataFrame({
              'foc': foc,
          })
          sns.displot(df, x="foc")
          plt.show()'''

        # if float(k[5])>6.5
        # dataSet = np.array(dataSet).astype(float)
        bl = np.array([k for k in dataSet])[:, 2]
        for i in range(0, len(bl)):
            bl[i] = 'L'
        dataSet[:, 2] = bl

        wd = np.array([k for k in dataSet])[:, 10]
        for i in range(0, len(wd)):
            if float(wd[i]) > 180:
                wd[i] = float(wd[i]) - 180  # and  float(k[8])<20
        dataSet[:, 10] = wd

        wf = np.array([k for k in dataSet])[:, 11]
        for i in range(0, len(wf)):

              wf[i] = self.ConvertMSToBeaufort(float(float(wf[i])))
        dataSet[:, 11] = wf

        lenConditionTlg = 5000000
        dtNew = np.array([k for k in dataSet if float(k[15]) > 0 and float(k[12]) > 0])  # and  float(k[8])<20
        # dtNewBDD = np.array([k for k in dataSetBDD if float(k[15]) > 0 and float(k[12]) > 0])
        # dtNewADD = np.array([k for k in dataSetADD if float(k[15]) > 0 and float(k[12]) > 0])

        # ballastDt = np.array([k for k in dtNew if k[2] == 'B' ])[:, 7:].astype(float)
        # ladenDt = np.array([k for k in dtNew if k[2] == 'L' ])[:, 7:].astype(float)
        draft =np.array([k for k in dataSet if float(k[8]) > 0])[:, 8].astype(float)
        meanDraft = np.mean(draft)
        stdDraft = np.std(draft)
        # meanDraftBallast = round(float(np.mean(np.array([k for k in ballastDt if k[1] > 0])[:, 1])), 2)
        # meanDraftLadden = round(float(np.mean(np.array([k for k in ladenDt if k[1] > 0])[:, 1])), 2)
        # minDraftLadden = round(float(np.min(np.array([k for k in ladenDt if k[1] > 0])[:, 1])), 2)
        # maxDraftBallast = round(float(np.max(np.array([k for k in ballastDt if k[1] > 0])[:, 1])), 2)

        '''for i in range(0, len(dtNew)):
              # tNew[i, 10] = self.getRelativeDirectionWeatherVessel(float(dtNew[i, 7]), float(dtNew[i, 10]))
              if str(dtNew[i, 2]) == 'nan' and float(dtNew[i, 8]) > 0:
                  if float(dtNew[i, 8])  >= meanDraft+stdDraft:
                      dtNew[i, 2] = 'L'
                  else:
                      # if float(dtNew[i, 8]) <=meanDraftBallast+1:
                      dtNew[i, 2] = 'B'''''
        ########################################################################
        ########################################################################
        ########################################################################
        ladenFlag = False
        ballastFlag = False

        ballastDt = np.array([k for k in dtNew if k[2] == 'B' and float(k[15]) > 1 ])
        if ballastDt.__len__() > 0:
            ballastDt = ballastDt[:, 7:].astype(float)
            ballastFlag = True

        ladenDt = np.array([k for k in dtNew if k[2] == 'L' and float(k[15]) > 1 ])

        if ladenDt.__len__() > 0 :
            ladenDt = ladenDt[:, 7:].astype(float)
            ladenFlag=True


        '''for i in range(0, len(ballastDt)):
              ballastDt[i,12] = 1 if ballastDt[i,12]==0 else 1
              ballastDt[i] = np.mean(ballastDt[i:i + 15], axis=0)'''
        #ladenDt = ladenDt[:20000]
        for i in range(0, len(ladenDt)):
              ladenDt[i, 12] = 1 if ladenDt[i, 12] == 0 else 1
              ladenDt[i] = np.mean(ladenDt[i:i + 15], axis=0)

        '''meanLaddenFoc = np.mean(ladenDt[:, 8])
        stdLadenFoc = np.std(ladenDt[:, 8])
        ladenDt = np.array([k for k in ladenDt if
                            k[8] >= meanLaddenFoc - (2 * stdLadenFoc) and k[8] <= meanLaddenFoc + (
                                    2 * stdLadenFoc)])'''
        ###################################################################################LADDEN BEST FIT
        if ladenFlag == True:


            speedFoc = np.array([k for k in ladenDt if k[5] > 7])

            # stw = speedFoc#[:10000,5]#.reshape(-1,1)
            foc = speedFoc[:, 8]  # .reshape(-1,1)

            minfoc = np.min(foc)
            maxfoc = np.max(foc)

            ranges = []
            k = 0
            i = minfoc if minfoc > 0 else 1

            focsPLot = []
            swellWavePlot = []

            while i <= 60:
                # workbook._sheets[sheet].insert_rows(k+27)

                focArray = np.array([k for k in speedFoc if float(k[8]) >= i and float(k[8]) <= i + 3])
                # focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')

                if focArray.__len__() > 0:
                    focsPLot.append(focArray.__len__())
                    swellWavePlot.append(np.round((np.mean(np.nan_to_num(focArray[:, 15].astype(float)))), 2))
                    ranges.append(i)
                i += 3
                k += 1

            xi = np.array([k for k in ladenDt if k[15] > 0 and k[15]<=1 if (k[5] > 7 and k[5]<=9)  ])[:,8]
            yi = np.array([k for k in ladenDt if k[15] > 1 and k[15] <= 2 if (k[5] > 7 and k[5]<=9) ])[:,8]
            zi = np.array([k for k in ladenDt if k[15] > 2 and k[15] <= 3 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            fi = np.array([k for k in ladenDt if k[15] > 3 and k[15] <= 4 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            ci = np.array([k for k in ladenDt if k[15] > 4 and k[15] <= 5 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            si = np.array([k for k in ladenDt if k[15] > 5 and k[15] <= 6 if (k[5] > 7 and k[5] <= 9)])[:, 8]

            weightsSWH79=[0,ks_2samp(xi,yi)[0]/10,ks_2samp(yi,zi)[0]/10,ks_2samp(zi,fi)[0]/10,ks_2samp(fi,ci)[0]/10,
                          ks_2samp(ci,si)[0]/10,ks_2samp(ci, si)[0] / 10,ks_2samp(ci, si)[0] / 10]

            xi = np.array([k for k in ladenDt if k[15] > 0 and k[15] <= 1 if (k[5] > 9 and k[5] <= 11)])[:, 8]
            yi = np.array([k for k in ladenDt if k[15] > 1 and k[15] <= 2 if (k[5] > 9 and k[5] <= 11)])[:, 8]
            zi = np.array([k for k in ladenDt if k[15] > 2 and k[15] <= 3 if (k[5] > 9 and k[5] <= 11)])[:, 8]
            fi = np.array([k for k in ladenDt if k[15] > 3 and k[15] <= 4 if (k[5] > 9 and k[5] <= 11)])[:, 8]
            ci = np.array([k for k in ladenDt if k[15] > 4 and k[15] <= 5 if (k[5] > 9 and k[5] <= 11)])[:, 8]
            si = np.array([k for k in ladenDt if k[15] > 5 and k[15] <= 6 if (k[5] > 9 and k[5] <= 11)])#
            si = si [:,8]  if si.__len__()>0 else ci

            weightsSWH911 = [0,ks_2samp(xi, yi)[0] / 10, ks_2samp(yi, zi)[0] / 10, ks_2samp(zi, fi)[0] / 10,
                            ks_2samp(fi, ci)[0] / 10, ks_2samp(ci, si)[0] / 10,ks_2samp(ci, si)[0] / 10,ks_2samp(ci, si)[0] / 10]

            xi = np.array([k for k in ladenDt if k[15] > 0 and k[15] <= 1 if (k[5] > 11 and k[5] <= 14)])[:, 8]
            yi = np.array([k for k in ladenDt if k[15] > 1 and k[15] <= 2 if (k[5] > 11 and k[5] <= 14)])[:, 8]
            zi = np.array([k for k in ladenDt if k[15] > 2 and k[15] <= 3 if (k[5] > 11 and k[5] <= 14)])[:, 8]
            fi = np.array([k for k in ladenDt if k[15] > 3 and k[15] <= 4 if (k[5] > 11 and k[5] <= 14)])[:, 8]
            ci = np.array([k for k in ladenDt if k[15] > 4 and k[15] <= 5 if (k[5] > 11 and k[5] <= 14)])[:, 8]
            si = np.array([k for k in ladenDt if k[15] > 5 and k[15] <= 6 if (k[5] > 11 and k[5] <= 14)])#[:, 8]
            si = si [:,8] if si.__len__() > 0 else ci

            weightsSWH1114 = [0,ks_2samp(xi, yi)[0] / 10, ks_2samp(yi, zi)[0] / 10, ks_2samp(zi, fi)[0] / 10,
                             ks_2samp(fi, ci)[0] / 10, ks_2samp(ci, si)[0] / 10, ks_2samp(ci, si)[0] / 10, ks_2samp(ci, si)[0] / 10]

            ############################################################################################################################
            ############################################################################################################################
            xi = np.array([k for k in ladenDt if k[4] > 0 and k[4] <= 1 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            yi = np.array([k for k in ladenDt if k[4] > 1 and k[4] <= 2 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            zi = np.array([k for k in ladenDt if k[4] > 2 and k[4] <= 3 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            fi = np.array([k for k in ladenDt if k[4] > 3 and k[4] <= 4 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            ci = np.array([k for k in ladenDt if k[4] > 4 and k[4] <= 5 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            si = np.array([k for k in ladenDt if k[4] > 5 and k[4] <= 6 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            di = np.array([k for k in ladenDt if k[4] > 6 and k[4] <= 7 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            ri = np.array([k for k in ladenDt if k[4] > 7 and k[4] <= 8 if (k[5] > 7 and k[5] <= 9)])[:, 8]

            weightsWS79 = [0, ks_2samp(xi, yi)[0] / 10, ks_2samp(yi, zi)[0] / 10, ks_2samp(zi, fi)[0] / 10,
                            ks_2samp(fi, ci)[0] / 10,
                            ks_2samp(ci, si)[0] / 10, ks_2samp(si, di)[0] / 10, ks_2samp(di, ri)[0] / 10]

            xi = np.array([k for k in ladenDt if k[4] > 0 and k[15] <= 1 if (k[5] > 9 and k[5] <= 11)])[:, 8]
            yi = np.array([k for k in ladenDt if k[4] > 1 and k[15] <= 2 if (k[5] > 9 and k[5] <= 11)])[:, 8]
            zi = np.array([k for k in ladenDt if k[4] > 2 and k[15] <= 3 if (k[5] > 9 and k[5] <= 11)])[:, 8]
            fi = np.array([k for k in ladenDt if k[4] > 3 and k[15] <= 4 if (k[5] > 9 and k[5] <= 11)])[:, 8]
            ci = np.array([k for k in ladenDt if k[4] > 4 and k[15] <= 5 if (k[5] > 9 and k[5] <= 11)])[:, 8]
            si = np.array([k for k in ladenDt if k[4] > 5 and k[15] <= 6 if (k[5] > 9 and k[5] <= 11)])  #
            si = si[:, 8] if si.__len__() > 0 else ci
            di = np.array([k for k in ladenDt if k[4] > 6 and k[4] <= 7 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            ri = np.array([k for k in ladenDt if k[4] > 7 and k[4] <= 8 if (k[5] > 7 and k[5] <= 9)])[:, 8]

            weightsWS911 = [0, ks_2samp(xi, yi)[0] / 10, ks_2samp(yi, zi)[0] / 10, ks_2samp(zi, fi)[0] / 10,
                             ks_2samp(fi, ci)[0] / 10, ks_2samp(ci, si)[0] / 10, ks_2samp(si, di)[0] / 10,
                             ks_2samp(di, ri)[0] / 10]

            xi = np.array([k for k in ladenDt if k[4] > 0 and k[15] <= 1 if (k[5] > 11 and k[5] <= 14)])[:, 8]
            yi = np.array([k for k in ladenDt if k[4] > 1 and k[15] <= 2 if (k[5] > 11 and k[5] <= 14)])[:, 8]
            zi = np.array([k for k in ladenDt if k[4] > 2 and k[15] <= 3 if (k[5] > 11 and k[5] <= 14)])[:, 8]
            fi = np.array([k for k in ladenDt if k[4] > 3 and k[15] <= 4 if (k[5] > 11 and k[5] <= 14)])[:, 8]
            ci = np.array([k for k in ladenDt if k[4] > 4 and k[15] <= 5 if (k[5] > 11 and k[5] <= 14)])[:, 8]
            si = np.array([k for k in ladenDt if k[4] > 5 and k[15] <= 6 if (k[5] > 11 and k[5] <= 14)])  # [:, 8]
            si = si[:, 8] if si.__len__() > 0 else ci
            di = np.array([k for k in ladenDt if k[4] > 6 and k[4] <= 7 if (k[5] > 7 and k[5] <= 9)])[:, 8]
            ri = np.array([k for k in ladenDt if k[4] > 7 and k[4] <= 8 if (k[5] > 7 and k[5] <= 9)])[:, 8]

            weightsWS1114 = [0, ks_2samp(xi, yi)[0] / 10, ks_2samp(yi, zi)[0] / 10, ks_2samp(zi, fi)[0] / 10,
                              ks_2samp(fi, ci)[0] / 10, ks_2samp(ci, si)[0] / 10, ks_2samp(si, di)[0] / 10,
                              ks_2samp(di, ri)[0] / 10]


            foc01 = [(itm, '0-1') for itm in xi]
            foc12 = [(itm, '1-2') for itm in yi]
            foc23 = [(itm, '2-3') for itm in zi]
            foc34 = [(itm, '3-4') for itm in fi]
            foc45 = [(itm, '4-5') for itm in ci]
            joinedFoc =  foc12 + foc23

            df = pd.DataFrame(data=joinedFoc,
                columns=['foc', 'swh'])
            #df.Zip = df.Zip.astype(str).str.zfill(5)
            sns.displot(df, x="foc", hue='swh')
            '''foc1 = yi
            df = pd.DataFrame({
                'foc1': foc1,
            })
            sns.displot(df, x="foc1",hue='swh')'''

            #plt.show()


            #xi = np.array(swellWavePlot)
            #yi = np.array(ranges)
            #zi = np.array(focsPLot)

            '''p2SWH = np.poly1d(np.polyfit(xi, yi, 1))
            xp = np.linspace(np.min(xi), np.max(xi), 100)

            plt.plot([], [], '.', xp, p2SWH(xp))

            plt.scatter(xi, yi, s=zi, c="red", alpha=0.4, linewidth=4)
            plt.xticks(np.arange(np.floor(min(xi)) - 1, np.ceil(max(xi)) + 1, 1))
            plt.yticks(np.arange(np.floor(min(yi)), np.ceil(max(yi)) + 1, 5))
            plt.xlabel("SWH (m)")
            plt.ylabel("foc")
            plt.show()'''
            X = 0

            ###############################################################################################
            ###############################################################################################


            speedFoc = np.array([k for k in ladenDt if k[5] > 0 and (k[4] >= 0 and k[4] <= 1) and k[8]>1 ])[:40000]

            # stw = speedFoc#[:10000,5]#.reshape(-1,1)
            foc = speedFoc[:, 8]  # .reshape(-1,1)

            minfoc = np.min(foc)
            maxfoc = np.max(foc)

            focsApp = []
            meanSpeeds = []
            stdSpeeds = []
            ranges = []
            k = 0
            i = minfoc if minfoc > 0 else 1

            focsPLot = []
            speedsPlot = []

            while i <= 48 :
                # workbook._sheets[sheet].insert_rows(k+27)

                focArray = np.array([k for k in speedFoc if float(k[8]) >= i and float(k[8]) <= i + 3])
                # focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')

                if focArray.__len__() > 1:
                    focsPLot.append(focArray.__len__())
                    speedsPlot.append(np.round((np.mean(np.nan_to_num(focArray[:, 5].astype(float)))), 2))
                    ranges.append(i)
                i += 3
                k += 1

            xi = np.array(speedsPlot)
            yi = np.array(ranges)
            zi = np.array(focsPLot)

            p2 = np.poly1d(np.polyfit(xi, yi, 2))
            xp = np.linspace(np.min(xi), np.max(xi), 100)
            plt.plot([], [], '.', xp, p2(xp))

            #############################################################################################
            #############################################################################################


        if ballastFlag==True:
            #############################################################################BALLAST BEST FIT
            #meanballastFoc = np.mean(ballastDt[:, 8])
            #stdballastFoc = np.std(ballastDt[:, 8])

            speedFoc = np.array([k for k in ballastDt if k[5] > 0 and k[4] >= 0 and k[4] <= 3])[:len(ballastDt)]

            # stw = speedFoc#[:10000,5]#.reshape(-1,1)
            foc = speedFoc[:, 8]  # .reshape(-1,1)

            minfoc = np.min(foc)
            maxfoc = np.max(foc)

            focsApp = []
            meanSpeeds = []
            stdSpeeds = []
            ranges = []
            k = 0
            i = minfoc if minfoc > 0 else 1

            focsPLot = []
            speedsPlot = []

            while i <= maxfoc:
                # workbook._sheets[sheet].insert_rows(k+27)

                focArray = np.array([k for k in speedFoc if float(k[8]) >= i and float(k[8]) <= i + 1])
                # focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')

                if focArray.__len__() > 1:
                    focsPLot.append(focArray.__len__())
                    speedsPlot.append(np.round((np.min(np.nan_to_num(focArray[:, 5].astype(float)))), 2))
                    ranges.append(i)
                i += 1
                k += 1

            xi = np.array(speedsPlot)
            yi = np.array(ranges)
            zi = np.array(focsPLot)

            p2 = np.poly1d(np.polyfit(xi, yi, 2))
            xp = np.linspace(np.min(xi), np.max(xi), 100)
            #plt.plot([], [], '.', xp, p2(xp))

            ###########################################################################################################
            speedFoc = np.array([k for k in ballastDt if k[5] > 0 ])[:len(ballastDt)]

            # stw = speedFoc#[:10000,5]#.reshape(-1,1)
            foc = speedFoc[:, 8]  # .reshape(-1,1)

            minfoc = np.min(foc)
            maxfoc = np.max(foc)


            ranges = []
            k = 0
            i = minfoc if minfoc > 0 else 1

            focsPLot = []
            swellWavePlot = []

            while i <= 60:
                # workbook._sheets[sheet].insert_rows(k+27)

                focArray = np.array([k for k in speedFoc if float(k[8]) >= i and float(k[8]) <= i + 10])
                # focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')

                if focArray.__len__() >0:
                    focsPLot.append(focArray.__len__())
                    swellWavePlot.append(np.round((np.min(np.nan_to_num(focArray[:, 13].astype(float)))), 2))
                    ranges.append(i)
                i += 10
                k += 1

            #xi = np.array([k for k in ballastDt if k[13] >= 0 and k[13]<=1 and k[5]>=7 and k[5]<=9])[:,8]
            #yi = np.array([k for k in ballastDt if k[13] > 1 and k[13] <= 2 and k[5] >= 7 and k[5] <= 9])[:,8]

            xi = np.array(swellWavePlot)
            yi = np.array(ranges)
            zi = np.array(focsPLot)

            p2SWH = np.poly1d(np.polyfit(xi, yi, 2))
            xp = np.linspace(np.min(xi), np.max(xi), 100)

            plt.plot([], [], '.', xp, p2SWH(xp))

            plt.scatter(xi, yi, s=zi, c="red", alpha=0.4, linewidth=4)
            plt.xticks(np.arange(np.floor(min(xi)) - 1, np.ceil(max(xi)) + 1, 1))
            plt.yticks(np.arange(np.floor(min(yi)), np.ceil(max(yi)) + 1, 5))
            plt.xlabel("SWH (m)")
            plt.ylabel("foc")
            plt.show()
            X=0

        # meanDraftBallast = round(float(np.mean(np.array([k for k in ballastDt if k[1] > 0])[:, 1])), 2)
        # meanDraftLadden = round(float(np.mean(np.array([k for k in ladenDt if k[1] > 0])[:, 1])), 2)
        # minDraftLadden = round(float(np.min(np.array([k for k in ladenDt if k[1] > 0])[:, 1])), 2)
        # maxDraftBallast = round(float(np.max(np.array([k for k in ballastDt if k[1] > 0])[:, 1])), 2)

        '''draft = (np.array((np.array([k for k in dtNew if float(k[8]) > 0 and float(k[8]) < 20])[:, 8])).astype(float))
        trim = (np.array((np.array([k for k in dtNew if float(k[17]) < 20])[:, 17])).astype(float))
        velocities = (
            np.array((np.array([k for k in dtNew if float(k[12]) > 0])[:, 12])).astype(float))  # and float(k[12]) < 18
        velocitiesTlg = (
            np.array((np.array([k for k in dtNew if float(k[18]) > 0 and float(k[18]) < 70])[:, 12])).astype(float))'''
        '''if tlgDataset==[]:
              tlgDataset = dtNew
              tlgDatasetBDD = dtNewBDD
              tlgDatasetADD = dtNewADD

              #velocitiesTlgBDD = (
                  #np.array((np.array([k for k in dtNewBDD if float(k[18]) > 0 ])[:, 12])).astype(float))
              #velocitiesTlgADD = (
                  #np.array((np.array([k for k in dtNewADD if float(k[18]) > 0])[:, 12])).astype(float))
          else:
              velocitiesTlg = (np.array((np.array([k for k in dtNew if float(k[18]) > 0 and float(k[18])<35 ])[:, 18])).astype(float)) #and float(k[12]) < 18'''

        '''dataModel = KMeans(n_clusters=4)
          velocities = velocities.reshape(-1, 1)
          dataModel.fit(velocities)
          # Extract centroid values
          centroids = dataModel.cluster_centers_
          velocitiesSorted = np.sort(centroids, axis=0)
          ################################################################################################
          #ballastDt = np.array([k for k in dtNew if k[2] == 'B'])[:, 7:].astype(float)
          #ladenDt = np.array([k for k in dtNew if k[2] == 'L'])[:, 7:].astype(float)

          #velocitiesB = np.array([k for k in ballastDt if k[5] > 6 and k[5] < 16])[:, 5]

          dataModel = KMeans(n_clusters=4)
          #velocitiesB = velocitiesB.reshape(-1, 1)
          dataModel.fit(velocitiesB)
          labels = dataModel.predict(velocitiesB)
          # Extract centroid values

          centroidsB = dataModel.cluster_centers_
          centroidsB = np.sort(centroidsB, axis=0)'''
        ##LOAD EXCEL
        workbook = load_workbook(filename=pathToexcel)

        '''wind = [0, 22.5, 67.5, 112.5, 157.5, 180]

          ###SEA STATE
          ballastMaxSeaSate = []
          for i in range(0, len(wind) - 1):
              arrayWF = np.array(
                  [k for k in ballastDt if k[3] >= wind[i] and k[3] <= wind[i + 1] and k[4] <= 9 and k[8] > 0])
              maxSS = np.max(arrayWF[:, 4]) if arrayWF.__len__() > 0 else 0
              ballastMaxSeaSate.append(round(maxSS, 2))

          for i in range(3, 8):
              workbook._sheets[3]['B' + str(i)] = ballastMaxSeaSate[i - 3]
          ####################################################
          laddenMaxSeaSate = []
          for i in range(0, len(wind) - 1):
              arrayWF = np.array(
                  [k for k in ladenDt if k[3] >= wind[i] and k[3] <= wind[i + 1] and k[4] <= 9 and k[8] > 0])
              maxSS = np.max(arrayWF[:, 4]) if arrayWF.__len__() > 0 else 0
              laddenMaxSeaSate.append(round(maxSS, 2))

          for i in range(12, 17):
              workbook._sheets[3]['B' + str(i)] = laddenMaxSeaSate[i - 12]
          ###############################
          ##end of sea state
          ###
          # meanDraftBallast = round(float(np.mean(np.array([k for k in ballastDt if k[1] > 0])[:, 1])), 2)
          workbook._sheets[2]['B2'] = np.floor(meanDraftBallast) + 0.5
          # meanDraftLadden = round(float(np.mean(np.array([k for k in ladenDt if k[1] > 0])[:, 1])), 2)
          workbook._sheets[1]['B2'] = np.ceil(meanDraftLadden) + 0.5

          ##WRITE DRAFTS TO FILE
          drafts = []
          drafts.append(np.floor(meanDraftBallast) + 0.5)
          drafts.append(np.ceil(meanDraftLadden) + 0.5)
          with open('./data/' + company + '/' + vessel + '/ListOfDrafts.csv', mode='w') as data:
              data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
              data_writer.writerow(
                  ['Draft'])
              for i in range(0, 2):
                  data_writer.writerow(
                      [drafts[i]])

          #############################################################################
          ##BALLAST
          ##delete ballast outliers
          # np.delete(ladenDt, [i for (i, v) in enumerate(ladenDt[:, 8]) if v < (
          # np.mean(ladenDt[:, 8]) - np.std(ladenDt[:, 8])) or v > np.mean(
          # ladenDt[:, 8]) + np.std(
          # ladenDt[:, 8])], 0)
          partitionsX = []
          partitionLabels = []
          # For each label
          for curLbl in np.unique(labels):
              # Create a partition for X using records with corresponding label equal to the current
              partitionsX.append(np.asarray(velocitiesB[labels == curLbl]))
              # Create a partition for Y using records with corresponding label equal to the current

              # Keep partition label to ascertain same order of results
              partitionLabels.append(curLbl)

          sorted = []
          initialX = len(partitionsX)
          initialXpart = partitionsX
          while sorted.__len__() < initialX:
              min = 100000000000
              for i in range(0, len(partitionsX)):
                  mean = np.mean(partitionsX[i])
                  if mean < min:
                      min = mean
                      minIndx = i
              sorted.append(partitionsX[minIndx])
              # partitionsX.remove(partitionsX[minIndx])
              partitionsX.pop(minIndx)'''

        ################################################################################################################

        #workbook = self.calculateExcelStatistics(workbook, dtNew, velocities, draft, trim, velocitiesTlg, rawData,
                                                 #company, vessel, tlgDataset, 'all')
        # workbook = self.calculateExcelStatistics(workbook, dtNewBDD, velocities, draft, trim, velocitiesTlgBDD, rawData,
        # company, vessel, tlgDatasetBDD,'bdd')
        # workbook = self.calculateExcelStatistics(workbook, dtNewADD, velocities, draft, trim, velocitiesTlgBDD, rawData,
        # company, vessel, tlgDatasetADD, 'Add')
        ##delete ladden outliers
        # np.delete(ladenDt, [i for (i, v) in enumerate(ladenDt[:, 8]) if v < (
        # np.mean(ladenDt[:, 8]) - np.std(ladenDt[:, 8])) or v > np.mean(
        # ladenDt[:, 8]) + np.std(
        # ladenDt[:, 8])], 0)

        minAccThres = 0

        if ballastFlag ==True:
            meanballastFoc = np.mean(ballastDt[:, 8])
            stdballastFoc = np.std(ballastDt[:, 8])
            #ballastDt = np.array([k for k in ballastDt if
                                #k[8] >= meanballastFoc - (2 * stdballastFoc) and k[8] <= meanballastFoc + (2 * stdballastFoc)])

            windForceWeights = [0, 0.25, 0.45, 0.65, 0.75, 1.15, 1.25, 1.35, 1.45]
            swellHeightWeights = weightsSWH1114
                #[0, 0.0043, 0.0023, 0.0024, 0.0025, 0.0046, 0.0057, 0.0058, 0.0059]

            ###########################################################################################
            foc0 = np.mean(np.array([k for k in ballastDt if k[3] > 0 and k[3] <= 22.5])[:, 8])
            foc1 = np.mean(np.array([k for k in ballastDt if k[3] > 22.5 and k[3] <= 67.5])[:, 8])
            foc2 = np.mean(np.array([k for k in ballastDt if k[3] > 67.5 and k[3] <= 112.5])[:, 8])
            foc3 = np.mean(np.array([k for k in ballastDt if k[3] > 112.5 and k[3] <= 157.5])[:, 8])
            foc4 = np.mean(np.array([k for k in ballastDt if k[3] > 157.5 and k[3] <= 180])[:, 8])

            '''wd0 = abs((foc1 - foc0) / foc0)
              wd1 = abs((foc2 - foc1) / foc1)
              wd2 = abs((foc3 - foc2) / foc2)
              wd3 = abs((foc4 - foc0) / foc4)'''

            wd0 = abs(1 / (1 + (foc4 - foc0)))  # against - with
            wd1 = abs(1 / (1 + (foc1 - foc3)))  # against side - side with
            wd2 = (abs(1 / (1 + (foc2 - foc4))))  # + abs(1 / (1 + (foc2 - foc4))))/2 #side - with - against
            wd3 = 0.0123  # side with - side
            wd4 = 0

            windDirWeights = [wd0, wd1, wd2, wd3, wd4]

            velMin = 11.75
            velMax = 12.25

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [641, 650, 659, 668, 677, 686, 695, 704]
            wind = [0, 22.5, 67.5, 112.5, 157.5, 180]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round((ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                    swellHeightWeights[s])) + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 7.75
            velMax = 8.25

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [10, 19, 28, 37, 46, 55, 64, 73]
            wind = [0, 22.5, 67.5, 112.5, 157.5, 180]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            speed = [7, 8]  # ,8.75,9.25,9.75]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

            ballastDt11_8 = []
            numberOfApp11_8 = []
            ballastDt7801 = []
            for w in range(0, len(windF) - 1):

                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round((ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                    swellHeightWeights[s])) + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")
                lastLenballastDt7801 = len(ballastDt7801)

            # workbook.save(filename=pathToexcel.split('.')[0] + '_1.' + pathToexcel.split('.')[1])
            # return
            ####################END 8 SPEED ########################################################################
            ####################END 8 SPEED ########################################################################

            velMin = 8.25
            velMax = 8.75

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if

            row = [89, 98, 107, 116, 124, 134, 143, 152]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round((ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                    swellHeightWeights[s])) + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            ##########################################################################################END SPEED 8.5
            ##########################################################################################END SPEED 8.5
            velMin = 8.75
            velMax = 9.25

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [168, 177, 186, 195, 204, 213, 222, 231]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []

            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                        swellHeightWeights[s])) +
                                    + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 9.25
            velMax = 9.75

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [247, 256, 265, 274, 283, 292, 301, 310]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                        swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 9.75
            velMax = 10.25

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [325, 334, 343, 352, 361, 370, 379, 388]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                        swellHeightWeights[s])) +
                                    + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 10.25
            velMax = 10.75

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [404, 413, 422, 431, 440, 449, 458, 467]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                        swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 10.75
            velMax = 11.25

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [483, 492, 501, 510, 519, 528, 537, 546]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                        swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 11.25
            velMax = 11.75

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [562, 571, 580, 589, 598, 607, 616, 626]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                        swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 12.25
            velMax = 12.75

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [720, 729, 738, 747, 756, 765, 774, 783]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                        swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 12.75
            velMax = 13.25

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [798, 807, 816, 825, 834, 843, 852, 861]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                        swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 13.25
            velMax = 13.75

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [877, 886, 895, 904, 913, 922, 931, 940]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                        swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 13.75
            velMax = 14.75

            FocCentral = np.array([k for k in ballastDt if
                                   k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ballastDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            meanFocCentral = np.mean(FocCentral)
            stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ballastDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [955, 964, 973, 982, 991, 1000, 1009, 1018]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ballastDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ballastDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        arrayFoc = np.array([k for k in ballastDt if
                                             k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                 5] <= velMax and k[3] >= wind[i] and
                                             k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ballastDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ballastDt7801[len(ballastDt7801) - 5] + ballastDt7801[len(ballastDt7801) - 5] * (
                                        swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ballastDt7801[lastLenballastDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ballastDt7_8.append(cellValue)
                        ballastDt7801.append(cellValue)
                    lastLenDt_8 = len(ballastDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ballastDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[2][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

        ####################################################LADDEN START ###################################################################
        ####################################################LADDEN START ###################################################################
        ####################################################LADDEN START ###################################################################
        if ladenFlag==True:


            windForceWeights = [0, 0.25, 0.45, 0.65, 0.75, 1.15, 1.25, 1.35, 1.45]
            swellHeightWeights = weightsSWH1114
                #[0, 0.0043, 0.0023, 0.0024, 0.0025, 0.0046, 0.0057, 0.0058, 0.0059]

            ###########################################################################################
            foc0 = np.mean(np.array([k for k in ladenDt if k[3] > 0 and k[3] <= 22.5])[:, 8])
            foc1 = np.mean(np.array([k for k in ladenDt if k[3] > 22.5 and k[3] <= 67.5])[:, 8])
            foc2 = np.mean(np.array([k for k in ladenDt if k[3] > 67.5 and k[3] <= 112.5])[:, 8])
            foc3 = np.mean(np.array([k for k in ladenDt if k[3] > 112.5 and k[3] <= 157.5])[:, 8])
            foc4 = np.mean(np.array([k for k in ladenDt if k[3] > 157.5 and k[3] <= 180])[:, 8])

            '''wd0 = abs((foc1 - foc0) / foc0)
              wd1 = abs((foc2 - foc1) / foc1)
              wd2 = abs((foc3 - foc2) / foc2)
              wd3 = abs((foc4 - foc0) / foc4)'''

            wd0 = abs(1 / (1 + (foc4 - foc0)))  # against - with
            wd1 = abs(1 / (1 + (foc1 - foc3)))  # against side - side with
            wd2 = wd1 -0.3  # + abs(1 / (1 + (foc2 - foc4))))/2 #side - with - against
            wd3 = 0.0123  # side with - side
            wd4 = 0

            windDirWeights = [wd0, wd1, wd2, wd3, wd4]

            velMin = 11.75
            velMax = 12.25

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            #meanFocCentral = np.mean(FocCentral)
            #stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [641, 650, 659, 668, 677, 686, 695, 704]
            wind = [0, 22.5, 67.5, 112.5, 157.5, 180]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            arrayFoc=[]
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round((ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 7.75
            velMax = 8.25

            swellHeightWeights = weightsSWH79

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [10, 19, 28, 37, 46, 55, 64, 73]
            wind = [0, 22.5, 67.5, 112.5, 157.5, 180]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            speed = [7, 8]  # ,8.75,9.25,9.75]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

            ladenDt11_8 = []
            numberOfApp11_8 = []
            ladenDt7801 = []
            for w in range(0, len(windF) - 1):

                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                             2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            #meanFoc = (weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round((ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")
                lastLenLadenDt7801 = len(ladenDt7801)

            # workbook.save(filename=pathToexcel.split('.')[0] + '_1.' + pathToexcel.split('.')[1])
            # return
            ####################END 8 SPEED ########################################################################
            ####################END 8 SPEED ########################################################################

            velMin = 8.25
            velMax = 8.75

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if

            row = [89, 98, 107, 116, 124, 134, 143, 152]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                             2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round((ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            ##########################################################################################END SPEED 8.5
            ##########################################################################################END SPEED 8.5
            velMin = 8.75
            velMax = 9.25

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]



            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [168, 177, 186, 195, 204, 213, 222, 231]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []

            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) +
                                    + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 9.25
            velMax = 9.75

            swellHeightWeights = weightsSWH911

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [247, 256, 265, 274, 283, 292, 301, 310]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ladenDt7801[len(ladenDt7801) - 5] +  ( swellHeightWeights[s])) + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 9.75
            velMax = 10.25

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [325, 334, 343, 352, 361, 370, 379, 388]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) +
                                    + windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 10.25
            velMax = 10.75

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [404, 413, 422, 431, 440, 449, 458, 467]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ladenDt7801[len(ladenDt7801) - 5] +  ( swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 10.75
            velMax = 11.25

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [483, 492, 501, 510, 519, 528, 537, 546]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 11.25
            velMax = 11.75
            swellHeightWeights = weightsSWH1114

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [562, 571, 580, 589, 598, 607, 616, 626]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 12.25
            velMax = 12.75

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [720, 729, 738, 747, 756, 765, 774, 783]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 12.75
            velMax = 13.25

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [798, 807, 816, 825, 834, 843, 852, 861]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 13.25
            velMax = 13.75

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [877, 886, 895, 904, 913, 922, 931, 940]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

            velMin = 13.75
            velMax = 14.75

            #FocCentral = np.array([k for k in ladenDt if
                                   #k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]


            # FocCentral = np.array([k for k in FocCentral if
            # (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral)])

            # steamTimeGen = np.array([k for k in ladenDt if
            # [5] >= velMin and k[5] <= velMax and (k[8] > 0 and (k >= meanFocCentral - 2 * stdFocCentral and k <= meanFocCentral + 2 * stdFocCentral))])[:, 12]

            # weighted_avgFocCentral = np.average(FocCentral, weights=steamTimeGen)

            # meanFocCentral = np.mean(FocCentral)
            # stdFocCentral = np.std(FocCentral)

            centralMean = p2(velMin )  # weighted_avgFocCentral
            # centralArray = np.array([k for k in ladenDt if
            # k[5] >= velMin and k[5] <= velMax and k[8] > 0])[:, 8]

            row = [955, 964, 973, 982, 991, 1000, 1009, 1018]
            windF = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            swellH = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            column = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            ladenDt7801 = []
            numberOfApp11_8 = []
            for w in range(0, len(windF) - 1):
                for s in range(0, len(swellH) - 1):
                    ladenDt7_8 = []
                    numberOfApp11_8 = []
                    for i in range(0, len(wind) - 1):

                        #arrayFoc = np.array([k for k in ladenDt if
                                             #k[4] > windF[w] and k[4] <= windF[w + 1] and k[5] > velMin and k[
                                                # 5] <= velMax and k[3] >= wind[i] and
                                             #k[3] <= wind[i + 1] and k[13] >= swellH[s] and k[13] <= swellH[s + 1]])
                        if arrayFoc.__len__() > minAccThres:
                            meanArrayFoc = np.mean(arrayFoc[:, 8])
                            stdArrayFoc = np.std(arrayFoc[:, 8])
                            arrayFoc = np.array([k for k in arrayFoc if
                                                 k[8] >= meanArrayFoc - (2 * stdArrayFoc) and k[8] <= meanArrayFoc + (
                                                         2 * stdArrayFoc)])

                            steamTime = arrayFoc[:, 12]

                        tlgarrayFoc = arrayFoc[:, 9] if arrayFoc.__len__() > minAccThres else []
                        tlgarrayFoc = np.array([k for k in tlgarrayFoc if k > 5])
                        tlgarrayFoc = np.array(
                            [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])
                        if tlgarrayFoc.__len__() > lenConditionTlg:
                            tlgarrayFoc = np.array(

                                [k for k in ladenDt if k[5] > velMax and k[5] <= velMax and k[8] > 10])[:, 9]
                            meanFoc = (np.mean(arrayFoc[:, 8]) + np.mean(
                                tlgarrayFoc) + centralMean) / 3 if arrayFoc.__len__() > minAccThres else (
                                                                                                                 centralMean + np.mean(
                                                                                                             tlgarrayFoc)) / 2
                            numberOfApp11_8.append(arrayFoc.__len__() + tlgarrayFoc.__len__() + centralArray.__len__())
                        else:
                            # np.average(arrayFoc[:, 8],weights=steamTime)
                            weighted_avgFocArray = np.average(arrayFoc[:, 8],
                                                              weights=steamTime) if arrayFoc.__len__() > minAccThres else centralMean
                            meanFoc = (
                                              weighted_avgFocArray + centralMean) / 2 if arrayFoc.__len__() > minAccThres else centralMean
                            numberOfApp11_8.append(arrayFoc.__len__())  # + centralArray.__len__())
                            if (s > 0 and w >= 0):
                                cellValue = round(
                                    (ladenDt7801[len(ladenDt7801) - 5] + ( swellHeightWeights[s])) +
                                    windDirWeights[i], 2)
                            elif s == 0 and w == 0:
                                cellValue = round(centralMean + windDirWeights[i], 2)
                            elif s == 0 and w > 0:
                                cellValue = round(centralMean + windForceWeights[w] + windDirWeights[i], 2)
                                # round(ladenDt7801[lastLenLadenDt7801 - (39-((i-1 if i < 5 else i-2) ))] + windForceWeights[w] + windDirWeights[i], 2)
                        ladenDt7_8.append(cellValue)
                        ladenDt7801.append(cellValue)
                    lastLenDt_8 = len(ladenDt7_8)
                    for i in range(row[w], row[w] + 5):
                        try:
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)] = str(
                                ladenDt7_8[i - row[w]])  # + '(' + str(numberOfApp11_8[i - row[w]]) + ')'
                            workbook._sheets[1][column[s - 1 if s == 8 else s] + str(i)].alignment = Alignment(
                                horizontal='right')
                        except:
                            print("Exception")

        workbook.save(filename=pathToexcel.split('.')[0] + '_1.' + pathToexcel.split('.')[1])
        return
        #################################
