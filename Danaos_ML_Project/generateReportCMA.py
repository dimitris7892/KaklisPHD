import argparse
import os
from typing import Optional, List, Any, AnyStr
import subprocess
import global_land_mask
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import geopy.distance
import shutil
import seaborn as sns
import cx_Oracle
from sklearn.cluster import KMeans
from matplotlib import pyplot
import locale
import plotly.graph_objects as go
from geographiclib.geodesic import Geodesic
from fastapi import APIRouter
import requests
from datetime import date, datetime, timedelta
import json
import csv
import pyproj
import dataReading as dRead
import neuralDT as NDT
import preProcessLegs as prePro
import PIL.ImageDraw
import plotly
import configparser

plotly.io.orca.config.mapbox_access_token = "pk.eyJ1IjoiZGltaXRyaXNrYWsiLCJhIjoiY2t1bDZjZHF6MGdmaTJvbW9pbHQ0cWlpMiJ9._4nRZ4fY8Ub1wcMj_MHdOw"
preLegs = prePro.preProcessLegs(knotsFlag=False)
dread = dRead.BaseSeriesReader()
geodesic = pyproj.Geod(ellps='WGS84')
locale.setlocale(locale.LC_ALL, '')
router = APIRouter()
config = configparser.ConfigParser()

class WaypointNew():
    Lon: float
    Lat: float
    PCode: str
    Pname: str
    Speed: float
    GC: bool
    Locked: bool
    WaitHR: float


class OnboardOutput():
    FoundSol: bool
    Crushed: bool
    ErrorMessage: str

    Path: List[WaypointNew]


class GenerateReport:

    def __init__(self, company, vessel, server, sid, password, usr):

        #self.config = config.read('configRepGen.ini')['DEFAULT']

        self.ndt = NDT.neuralDT(vessel)

        self.pathToWebAPIForPavlosInterpolation = \
            './danaos-deep-learning/ConsModelComparison/ConsModelComparison_webapi/bin/Debug/net5.0'

        self.pathToWebAPIForPavlosBreakRoute = \
            './danaos-deep-learning/ConsModelComparison/RouteBreakGen_webapi/bin/Debug/net5.0'

        self.requestJSON = {
            "client_request_id": 0,
            "vessel_code": "",
            "user_id": None,
            "optimization_model": {
                "vessel": {
                    "vslCode": "",
                    "laden": False,
                    "draft": 12.0,
                    "trim": 0.0
                },
                "voyage": {
                    "departDT": "2021-09-20T06:00:00",
                    "arriveDT": "2021-09-20T06:00:00",
                    "path": [

                    ],
                    "restrictETA": False,
                    "minSpeedVoyage": 0.0,
                    "maxSpeedVoyage": 20.0
                },
                "options": {
                    "fuel": False,
                    "ls": 0.0,
                    "hs": 0.0,
                    "time": False,
                    "tce": 0.0,
                    "weather": True
                },
                "bp": {
                    "openCanalsBP": [
                        "AKA",
                        "AMC",
                        "AMU",
                        "MOL",
                        "PAS",
                        "ROM",
                        "SEL",
                        "ATT",
                        "BAS",
                        "BAC",
                        "BIS",
                        "BNS",
                        "BEW",
                        "BOS",
                        "BOT",
                        "BUN",
                        "CCA",
                        "CFE",
                        "CHO",
                        "CGH",
                        "CRA",
                        "CWR",
                        "DAR",
                        "DOV",
                        "FAI",
                        "FAS",
                        "FIN",
                        "FLO",
                        "GIB",
                        "GBR",
                        "HIN",
                        "HAS",
                        "INI",
                        "IDT",
                        "JOM",
                        "KAN",
                        "KOR",
                        "LBT",
                        "LOM",
                        "LON",
                        "MGA",
                        "MIS",
                        "MIT",
                        "MON",
                        "NCA",
                        "NSI",
                        "OST",
                        "OSU",
                        "PAN",
                        "PEN",
                        "QUO",
                        "RSS",
                        "DUR",
                        "RIA",
                        "SAG",
                        "SIF",
                        "SIE",
                        "SIW",
                        "SKA",
                        "SOM",
                        "SCA",
                        "STJ",
                        "SUZ",
                        "SUN",
                        "SAI",
                        "SAO",
                        "AFR",
                        "SOU",
                        "THU",
                        "TRI",
                        "TUG",
                        "TUS",
                        "UNI",
                        "USH",
                        "VOK",
                        "WOI",
                        "WIN",
                        "YUC",
                        "WOT"
                    ],
                    "closeCanalsBP": [
                        "ASB",
                        "ASD",
                        "ASG",
                        "BAH",
                        "ASK",
                        "NIB",
                        "BON",
                        "CCN",
                        "CCS",
                        "CHL",
                        "CDE",
                        "CDW",
                        "COR",
                        "NIG",
                        "NIH",
                        "KIE",
                        "MAA",
                        "MAG",
                        "MES",
                        "NIM",
                        "NIA",
                        "NIN",
                        "NCI",
                        "N1F",
                        "N1I",
                        "N1R",
                        "PRO",
                        "STE",
                        "TER",
                        "NIT",
                        "VIR",
                        "WOG",
                        "WLI"

                    ],
                    "antipiracy": False,
                    "envnavreg": True
                },
                "wres": {
                    "avoidWind": False,
                    "avoidVis": False,
                    "avoidComb": False,
                    "avoidSwell": False,
                    "avoidWaves": False,
                    "avoidIceline": False,
                    "windKnLimit": 0.0,
                    "windMSLimit": 0.0,
                    "visKmLimit": 0.0,
                    "combFL": 0.0,
                    "combQR": 0.0,
                    "combBM": 0.0,
                    "combBW": 0.0,
                    "combHD": 0.0,
                    "swellFL": 0.0,
                    "swellQR": 0.0,
                    "swellBM": 0.0,
                    "swellBW": 0.0,
                    "swellHD": 0.0,
                    "wavesFL": 0.0,
                    "wavesQR": 0.0,
                    "wavesBM": 0.0,
                    "wavesBW": 0.0,
                    "wavesHD": 0.0,
                    "iceCoverPercAllowed": 0.0
                },
                "nogoareas": []
            }
        }

        self.server = server
        self.sid = sid
        self.password = password
        self.usr = usr
        self.vessel = vessel
        self.company = company

        if self.vessel.__contains__(" "):

            self.vslFig = self.vessel.split(" ")[0] + "_" + self.vessel.split(" ")[1]

        else:
            self.vslFig = self.vessel

        if os.path.isdir('./Figures/' + self.company + '/' + self.vslFig) == False:
            os.mkdir('./Figures/' + self.company + '/' + self.vslFig)

        self.weatherCompTemplate = r'''
                        \section{Weather Comparison - Sensor / Weather Service (NOA) - Total }

                                    \begin{figure}[H]
                                        \centering
                                        \includegraphics
                                        [width=1\textwidth]                             
                                        {./Figures/%(company)s/%(vessel)s/windSpeedWS_NOA.pdf}
                                        \caption{\footnotesize{Wind Speed Sensor VS NOA Distributions comparison }}
                                        \label{fig:ws}
                                    \end{figure}                                     


                                   \begin{figure}[H]
                                    \centering
                                        \includegraphics
                                        [width=1\textwidth]                             
                                        {./Figures/%(company)s/%(vessel)s/windSpeedCompPerSpeed.eps}
                                        \caption{\footnotesize{Initial / Weather Features Comparison per Speed Range }}
                                        \label{fig:ws}
                                    \end{figure}
                                    '''

        self.templateLegInfo = r'''\
                                    \section{ %(leg)s }
                                     
                                    \begin{itemize}
                                        \item Total Actual FOC : \textbf{ %(totalActFoc)s MT} 
                                        
                                        \begin{itemize}
                                            \footnotesize
                                            {
                                                \item Total Predicted FOC (\textcolor{Blue}{Neural Net}): \textbf{%(totalPredFoc)s MT}
                                               
                                                \item FOC Perc Diff (\textcolor{Blue}{Neural Net}): \textbf{%(percDiff)4.2f %(perc)s}
                                              
                                            }
                                         \end{itemize}
                                         
                                        \item Departure Port : \textbf{ %(depPort)s / %(depDate)s}
                                        \item Arrival Port : \textbf{ %(arrPort)s / %(arrDate)s}
                                        \item Total Sailing Time: \textbf{ %(totalTime)s hours}
                                        \item Distance Travelled: \textbf{%(totalDist)s n mi}
                                        \item Draft: \textbf{ %(meanDraft)4.1f m }
                                    \end{itemize}
                                    
                                    
                                    \begin{figure}[H]
                                    \centering
                                    \includegraphics
                                    [width=1\textwidth]                             
                                    {./Figures/%(company)s/%(vessel)s/leg%(legNumberFig)s.eps}
                                    \caption{\footnotesize{FOC Actual vs Predicted / Speed Ranges, %(leg)s }}
                                    \label{fig:heatmap}
                                    \end{figure}
                                    
                                    \newpage
                                    \textbf{Weather Comparison - Sensor / Weather Service (NOA) }
                                    
                                                                                    
                                   \begin{figure}[H]
                                    \centering
                                    \includegraphics
                                    [width=1\textwidth]                             
                                    {./Figures/%(company)s/%(vessel)s/windSpeedWS_NOAleg%(legNumberFig)s.pdf}
                                    \caption{\footnotesize{Wind Speed Sensor VS NOA Distributions comparison }}
                                    \label{fig:ws}
                                    \end{figure}
                                    
                                    
                                    
                                     \begin{figure}[H]
                                    \centering
                                    \includegraphics
                                    [width=1\textwidth]                             
                                    {./Figures/%(company)s/%(vessel)s/windSpeedCompPerSpeedleg%(legNumberFig)s.eps}
                                    \caption{\footnotesize{Initial / Weather Features Comparison per Speed Range }}
                                    \label{fig:ws}
                                    \end{figure}
                                    
                                     \newpage
                                    
                                    %(templateROLegInfo)s

                                    \newpage
                                    '''

        self.title = r'''
                            \newgeometry{left= 2.6cm, right= 2.6cm, bottom= 2.5cm, top= 2.5cm} %% Modifiziert die Seitenabstände für die Titelseite

                            \begin{titlepage}
                            \begin{center}

                            \large{Danaos Management Consultants} \\

                            \vspace{1cm}
                            \noindent\rule{\textwidth}{1.5pt}
                            \vspace{0.3cm}

                            \LARGE{\textbf{ %%(vessel)s }} \\ %% Bei Masterarbeit hier den Text austauschen

                            \large{\today}\\

                            \vspace{1cm}


                            \vspace{0.3cm}
                            \noindent\rule{\textwidth}{1.5pt}
                            \vspace{1cm}
                            \end{center}

                            \noindent\normalsize{\textbf{Vessel Basic Info:}} \hfill \normalsize{} 

                            \vspace{0.5cm}

                            \noindent\normalsize{} \hfill \normalsize{} \\
                            \normalsize{IMO: %(imo)s } \\
                            \normalsize{Vessel Type: %(vesselType)s } \\
                            \normalsize{Deadweight: } \\
                            \normalsize{Beam: } \\
                            \normalsize{Length:  }\\
                            \normalsize{Min Draft: 6 m} \\
                            \normalsize{Max Draft: 12 m} \\
                            \normalsize{Max Speed: Sea speed Ballast . Laden } \\
                            \normalsize{
                            Dates of last two propeller cleanings:	 }\\
                            \normalsize{Dates of last two dry docks:  }\\
                            \normalsize{Voyages the vessel most commonly
                            Performs: }



                            \end{titlepage}

                            \restoregeometry %% Stellt Seitenabstände wieder her'''

        self.overallAcc = r'''  \textbf{{\large Overall Accuracy}}

                                \begin{itemize}
                                    \item Total records: \textbf{%(totalInstances)s instances}
                                    \item Total Sailing Time: \textbf{ %(totalTime)s hours}
                                    \item Total Distance Travelled: \textbf{ %(totalDist)s n mi}                                    
                                \end{itemize}


                                \begin{figure}[H]
                                \centering
                                \includegraphics
                                [width=1\textwidth] 
                                {./Figures/%(company)s/%(vslFig)s/overallAcc.eps}
                                \caption{\footnotesize{FOC Actual vs Predicted / Speed Ranges, Overall on Legs}}
                                \label{fig:heatmap}
                                \end{figure}


                        

                                \newpage'''

        self.templateSummaryTable = r'''
                            \captionof{table}{Neural Network model accuracy based on the weather observed by the sensors} 
                            \begin{table}[H]
                                \centering
                                 
                                    \label{tab:mae}
                                \resizebox{\textwidth}{!}{    
                                     \begin{tabular}{|c|c|c|c|c|c|}
                                    \hline
                                \multirow{1}{*}{} & \multicolumn{5}{|c|}{\textbf{Info per leg}}

                                \\

                                \cline{2-6}



                                        \textbf{ \large{%(model)s} } 
                                     &  \textbf{Total Act FOC (MT)} & 

                                     \textbf{Total Pred FOC (MT)} 

                                     & \textbf{FOC Perc Diff} & 

                                     \textbf{Mean STW (kt)}

                                     &  \textbf{Mean Draft (m)} \\

                                     \hline 

                                     \hline

                                       %(templateRowTable)s                              

                                    \hline 
                                    \textbf{Total } &  \multicolumn{1}{|c|
                                    }{\textbf{%(totalAvgActFoc)s }} &
                                    \multicolumn{1}{|c|}{\textbf{ %(totalAvgPredFoc)s }}  & 
                                    \multicolumn{1}{|c|}{\textbf{{ %(totalAvgPercDiff)4.2f %(perc)s}}} &
                                    \multicolumn{1}{|c|}{\textbf{ %(totalAvgSTW)4.1f (Avg)} } &
                                    \multicolumn{1}{|c|}{\textbf{ %(totalAvgDraft)4.1f (Avg)}}\\ 
                                    \hline 

                                    \end{tabular}
                                }

                                \end{table}

                            '''

        self.templateRowTable = r'''\hline 
                                %(leg)s &
                                %(totalActFoc)s  & %(totalPredFoc)s  & 
                                 %(percDiff)4.2f %(perc)s  &  %(meanSTW)4.2f  & %(meanDraft)4.2f \\  
                                 \hline'''

        self.opt_init_templateTrue = r'''


                                         \begin{figure}[H]
                                                \centering
                                                \includegraphics
                                                [width=1\textwidth]                             
                                                {./Figures/%(company)s/%(vessel)s/optimized_initial_True_%(legNumber)s.eps}
                                                \caption{\footnotesize{Wind Speed comparison Initial / Optimized Route, %(leg)s }}
                                                \label{fig:heatmap}
                                         \end{figure}
                                        '''

        self.opt_init_templateFalse = r'''

                                    \begin{figure}[H]
                                                      \centering
                                                      \includegraphics
                                                      [width=1\textwidth]                             
                                                      {./Figures/%(company)s/%(vessel)s/optimized_initial_False_%(legNumber)s.eps}
                                                      \caption{\footnotesize{Wind Speed comparison Initial / Optimized Route, %(leg)s }}
                                                      \label{fig:heatmap}
                                                \end{figure}
                                              '''

        self.templateROLegInfo = r'''
                                     \textbf{ Routing Optimization Info : \textbf{%(leg)s}}

                                    \begin{figure}[H]
                                        \centering
                                        \includegraphics
                                        [width=1\textwidth]                             
                                        {./Figures/%(company)s/%(vessel)s/map_%(legFig)s.pdf}
                                        \caption{\footnotesize{Initial / Optimized Route, %(leg)s }}
                                        \label{fig:heatmap}
                                    \end{figure}



                                    \begin{table}[H]
                                       \resizebox{\textwidth}{!}{ %(perc)s
                                            \begin{tabular}{|c|c|c|c|}

                                            \hline
                                            \textbf{Voyage} &
                                             \textbf{Date} &
                                            \textbf{Latitude} & \multicolumn{1}{c|}{\textbf{Longitude}} \\ \hline
                                            Departure      &  %(depDate)s  & %(latDep)s  & %(lonDep)s  \\ \hline
                                             Arrival        & %(arrDate)s    & %(latArr)s & %(lonArr)s\\ \hline
                                                           &     &   \multicolumn{2}{c|}{}   \\ \hline
                                          \textbf{BASIC COMPARISON} &
                                              \textbf{Actual route estimation} &
                                               \multicolumn{2}{c|}{\textbf{Optimized route estimation}} \\
                                                \hline
                                             Distance (nm)  & %(distNMEst)s  &  \multicolumn{2}{c|}{%(distNMOptTrue)s}   \\ \hline
                                             Time (hours)   &  %(totalTime)s    & \multicolumn{2}{c|}{%(totalTimeOptTrue)s}    \\ \hline
                                             Avg Speed (kt) &     %(avgSpeed)s&  \multicolumn{2}{c|}{  %(avgSpeedOptTrue)s }   \\ \hline
                                             Total FOC (MT) &    %(totalFocEst)s  &  \multicolumn{2}{c|}{%(totalFocOptTrue)s}      \\ \hline

                                            \end{tabular}
                                             
                                        }
                                        
                                       \caption{Estimation based on Weather Service data (NOA)}
                                        \end{table}
                                        
                                        
                                        

                                \newpage

                                \textbf{ Weather comparison optimization / initial:}


                                 


                                    \begin{figure}[H]
                                                \centering
                                                \includegraphics
                                                [width=%(scaleTrue)s\textwidth]                             
                                                {./Figures/%(company)s/%(vessel)s/optimized_initial_True_%(legNumber)s.eps}
                                                \caption{\footnotesize{Wind Speed comparison Initial / Optimized Route, %(leg)s }}
                                                \label{fig:heatmap}
                                    \end{figure}


                                
     

                                    '''

        self.main = r'''
            \documentclass[a4paper,12pt,dvipsnames]{scrartcl}


            \usepackage[
                pdftitle={},
                pdfsubject={},
                pdfauthor={},
                pdfkeywords={},
                pdftex=true,
                colorlinks=true,
                breaklinks=true,
                citecolor=black,
                linkcolor=black,
                menucolor=black,
                urlcolor=black
            ]{hyperref}


            \usepackage[utf8]{inputenc}
            \usepackage[T1]{fontenc}   			


            \usepackage{subfigure}
            \usepackage[subfigure]{tocloft}
            \usepackage{hyperref}
            
            \usepackage{mdwlist}
            \usepackage{todonotes} 
            

            \usepackage[normalem]{ulem}
            \usepackage[autostyle=true,english=american]{csquotes} 
            \renewcommand{\mkblockquote}[4]{\enquote{#1}#2\ifterm{\relax}{#3}#4} 
            \renewcommand{\mkcitation}[1]{#1} 
            \usepackage{ragged2e} 

            \interfootnotelinepenalty10000  


            \usepackage{setspace} %% Package zur Veränderung des Zeilenabstandes innerhalb des Textes
            \expandafter\def\expandafter\quotation\expandafter{\quote\singlespacing} %% Einfacher Zeilenabstand in langen Zitaten
            \onehalfspacing %% 1,5-facher Zeilenabstand im gesamten Dokument
            %%\renewcommand{\baselinestretch}{1.5} %% Zeilenabstand 
            %%\setlength{\footnotesep}{10.0pt} %% Abstand zwischen Fußnote und Text 

            %% Schusterjungen und Hurenkinder vermeiden
            \clubpenalty = 10000
            \widowpenalty = 10000 
            \displaywidowpenalty = 10000

            %% Mathe-Zeugs
            \usepackage{amsmath,amsfonts,amssymb} %% Package für mathematische Umgebungen (wie Formelgestaltungen, etc.) 
            \usepackage{array} %% Package zur Erzeugung von Matrizen (legt Position und Spalten fest)
            \usepackage{textcomp} %% Package zur Generierung zusätzlicher Symbolzeichen 

            %% Farben
            \usepackage{xcolor}

            %% Grafiken
            \usepackage{graphicx} %% Grafiken einfügen
        
            \usepackage{float} %% Package zur Festlegung der Position von Tabellen/Abbildungen
            \usepackage[clockwise]{rotating} %% Drehung von Bildern & Tabellen

            %% TikZ
            \usepackage{tikz}
            \usepackage{subfigure}

            %% Packages für das Erstellen von Tabellen
            \usepackage{tabularx} %% Package zur Gestaltung von Tabellen (Festlegen einer Tabellenbreite, automatischen Zeilenumbruch,...)
            \usepackage{booktabs} %% Das Hauptaugenmerk von booktabs liegt dabei auf der Gestaltung der horizontalen Linien innerhalb einer Tabelle.  
            \usepackage{multirow} %% Package, dass die Zusammenführung von Zellen in einer Tabelle ermöglicht
            \newcolumntype{L}[1]{>{\RaggedRight\arraybackslash}p{#1}}
            \newcolumntype{C}[1]{>{\Centering\arraybackslash}p{#1}}
            \newcolumntype{R}[1]{>{\RaggedLeft\arraybackslash}p{#1}}

            %% Bild- & Tabellenunterschriften & -quellen
            \usepackage{caption}
            \captionsetup{format=plain,justification=RaggedRight,singlelinecheck=false}
            
            \usepackage[labelfont=bf]{caption}			
            \captionsetup[table]{justification=centerlast}

            %% Seitenformatierung
            \usepackage[left= 3cm, right= 2cm, bottom= 2.5cm, top= 2.5cm]{geometry} %% Seitenabstände
            \usepackage{pdflscape} %% Package zur Drehung von Seiten

            %% Überschriften-Schriftgröße und -art ändern
            \addtokomafont{disposition}{\rmfamily}
            %%\setkomafont{chapter}{\rmfamily\LARGE\bfseries} %% Nicht notwendig für die Klasse 'scrartcl'
            \setkomafont{section}{\rmfamily\Large\bfseries}
            \setkomafont{subsection}{\rmfamily\large\bfseries}

            %% Deutsches Datumsformat
            \usepackage{datetime}
            \newdateformat{myformat}{\THEDAY{. }\monthname[\THEMONTH] \THEYEAR}

            %% Kopf- & Fußzeile
            %%\usepackage{fancyhdr}
            %%\pagestyle{fancy} %% Muss vor \renewcommand{\chaptermark} stehen
            %%\fancyhf{} %% Bereinigt Kopf- & Fußzeile 
            %%\fancyfoot[L]{} %% Links
            %%\fancyfoot[C]{\thepage} %% Mitte
            %%\fancyfoot[R]{} %% Rechts
            %%\renewcommand{\chaptermark}[1]{
            %%	\markboth{Kapitel \thechapter{}: #1}{}}
            %%\renewcommand{\sectionmark}[1]{
            %%	\markright{\thesection{} #1}}
            %%\renewcommand{\headrulewidth}{0pt} %% Dicke der Trennlinie oben

            %% Tabellen- und Abbildungsverzeichnis nach der Lehrstuhlvorlage
            \usepackage{tocloft}
            \renewcommand{\cftfigpresnum}{Figure }
            \renewcommand{\cfttabpresnum}{Table }

            \renewcommand{\cftfigaftersnum}{:}
            \renewcommand{\cfttabaftersnum}{:}

            \setlength{\cftfignumwidth}{3cm}
            \setlength{\cfttabnumwidth}{2,5cm}

            \setlength{\cftfigindent}{0,5cm}
            \setlength{\cfttabindent}{0,5cm}

            %%  Zitation
            \usepackage[authordate,backend=biber, doi=false, isbn=false, footmarkoff]{biblatex-chicago}
            %% Hinzufügen der Literatur
            %%\addbibresource{Literatur.bib}


            %%%%%%%%%%%%%%%% Beginn Dokumentinformationen %%%%%%%%%%%%%%%%

            %%========== Dokumentbeginn ==========
            \begin{document}


            %% Titelseite
             \newgeometry{left = 2.6cm, right= 2.6cm, bottom= 2.5cm, top= 2.5cm} %% Modifiziert die Seitenabstände für die Titelseite


                            \begin{titlepage}
                            \begin{center}

                            \large{Danaos Management Consultants} \\

                            \vspace{1cm}
                            \noindent\rule{\textwidth}{1.5pt}
                            \vspace{0.3cm}

                            \LARGE{\textbf{ "%(vessel)s" }} \\ %% Bei Masterarbeit hier den Text austauschen

                            \large{\today}\\

                            \vspace{1cm}


                            \vspace{0.3cm}
                            \noindent\rule{\textwidth}{1.5pt}
                            \vspace{1cm}
                            \end{center}

                            \noindent\normalsize{\textbf{Vessel Basic Info:}} \hfill \normalsize{} 

                            \vspace{0.5cm}

                            \noindent\normalsize{} \hfill \normalsize{} \\
                            \normalsize{IMO: %(imo)s } \\
                            \normalsize{Vessel Type: Container} \\                  
                            \normalsize{Min Draft: %(minDraft)s m} \\
                            \normalsize{Max Draft: %(maxDraft)s m} \\
                            \normalsize{Max Speed: Sea speed Ballast %(maxBalSpeed)4.1f. Laden %(maxLadSpeed)4.1f} \\
                            
                          



                            \end{titlepage}

                            \restoregeometry %% Stellt Seitenabstände wieder her


            %% Inhaltsverzeichnis
            \newpage %% Erzeugt eine neue Seite
            \pagenumbering{Roman} %% Umschalten auf römische Seitenzahlnummerierung 
            %%\tableofcontents %% Erzeugt Inhaltsverzeichnis

            %% Abbildungsverzeichnis
            \newpage %% Erzeugt eine neue Seite
            \renewcommand\listfigurename{List Of Figures}
            %%\listoffigures
            %%\addcontentsline{toc}{section}{List Of Figures}

            %% Tabellenverzeichnis
            %%\newpage %% Erzeugt eine neue Seite
            %%\renewcommand\listtablename{List of Tables}
            %%\listoftables
            %%\addcontentsline{toc}{section}{List of Tables}


            %% Haupttext
            \newpage %% Erzeugt eine neue Seite
            \pagenumbering{arabic} %% Umschalten auf arabische Seitenzahlen
            \setcounter{page}{1} %% Setzt die Seitenzahl auf 1 zurück		

             \section{Model Performance Overview}

                %(templateSummaryTableNNet)s
                \raggedbottom   

               \vspace{1cm}


                
                \raggedbottom

                \newpage

                %(templateOvAcc)s

            \newpage


                %(templateLegInfo)s

            %% Literaturverzeichnis
            \newpage %% Erzeugt eine neue Seite
            \addcontentsline{toc}{section}{Literaturverzeichnis}
            \printbibliography[heading=bibliography]

            %% Anhang
            \newpage %% Erzeugt eine neue Seite
            \addcontentsline{toc}{section}{Anhang}


            %% Eidesstattliche Erklärung (nicht notwendig bei Seminar- und Hausarbeiten)
            \newpage %% Erzeugt eine neue Seite




            \end{document}
            %%========== Dokumentende ==========
                    '''

    def valCheck(self):

        path1 = './legs/' + self.vessel + '/'
        path2 = './correctedLegsForTR/' + self.vessel + '/'
        path3 = './sensAnalysis/' + self.vessel + '/'
        path4 = './sensAnalysisNeuralDT/' + self.vessel + '/'

        if len(sorted(glob.glob(path1 + '*.csv'))) == len(sorted(glob.glob(path2 + '*.csv'))) == len(
                sorted(glob.glob(path3 + '*.csv'))) \
                == len(sorted(glob.glob(path4 + '*.csv'))):

            return 1
        else:
            return -1

    def extractImoType(self, ):

        dsn = cx_Oracle.makedsn(
            self.server,
            '1521',
            service_name=self.sid
        )
        connection = cx_Oracle.connect(
            user=self.usr,
            password=self.password,
            dsn=dsn
        )
        cursor_myserver = connection.cursor()
        cursor_myserver.execute(
            'SELECT IMO_NO FROM VESSEL_DATA WHERE VESSEL_CODE =  (SELECT VESSEL_CODE FROM VESSEL_DATA WHERE VESSEL_NAME LIKE '"'%" + self.vessel + "%'"'  AND ROWNUM=1)  ')

        for row in cursor_myserver.fetchall():
            self.imo = row[0]
            print(str(self.imo))

        cursor_myserver.execute(
            'SELECT VESSEL_TYPE FROM VESSEL_DATA WHERE VESSEL_CODE =  (SELECT VESSEL_CODE FROM VESSEL_DATA WHERE VESSEL_NAME LIKE '"'%" + self.vessel + "%'"'  AND ROWNUM=1)  ')

        for row in cursor_myserver.fetchall():
            type = row[0]

        cursor_myserver.execute(
            'SELECT VESSEL_NAME FROM VESSEL_DATA WHERE VESSEL_CODE =  (SELECT VESSEL_CODE FROM VESSEL_DATA WHERE VESSEL_NAME LIKE '"'%" + self.vessel + "%'"'  AND ROWNUM=1)  ')

        for row in cursor_myserver.fetchall():
            orVslName = row[0]

            print(orVslName)

        return self.imo, type, orVslName

    def getListOfDataLegs(self, vessel):

        path = './correctedLegsForTR/' + vessel + '/'
        legsCorr = []
        sortedCorrLegs = self.sortLegsInFiles(path)
        for infile in sortedCorrLegs:
            print(infile)
            dataLegs = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values
            #print(len(dataLegs))
            #if len(dataLegs)>1000:
            legsCorr.append(dataLegs)

        path = './legs/' + vessel + '/'
        legs = []
        sortedLegsCorr = self.sortLegsInFiles(path)
        for infile in sortedLegsCorr:
            if os.path.isfile('./correctedLegsForTR/' + vessel + '/' + infile.split("/")[3]) == False:
                os.remove('./legs/' + vessel + '/' + infile.split("/")[3])
                continue
            dataLegs = pd.read_csv(infile, sep=',', decimal='.', skiprows=0, ).values
            #if len(dataLegs) < 1000: continue
            legs.append(dataLegs)

        return legsCorr, legs

    def sortLegsInFiles(self, path):

        # path = './legs/'+self.vessel+'/'

        files = []
        for infile in sorted(glob.glob(path + '*.csv')):
            #print("Sorting . . . " + infile)
            dataLeg = pd.read_csv(infile, sep=',', decimal='.')
            #if len(dataLeg.values)<1000:
                #continue
            #print(dataLeg['dt'])# .values
            files.append(infile + '_DT' + dataLeg['dt'][0])

        sorted_files = sorted(files, key=lambda x: x.split('_DT')[1])

        sorted_files = [k.split("_DT")[0] for k in sorted_files]
        return sorted_files

    def generate(self):

        imo, vslType, orVslName = self.extractImoType()

        dataListLegsCorr, dataListLegs = self.getListOfDataLegs(self.vessel)

        print(len(dataListLegs))
        print("LEN CORR :   "+str(len(dataListLegsCorr)))

        predDfs, accs, legs, sumActs, sumPreds, percDiffs, stws, drfts, latlonDep, latlonArr, sovgs = self.extractStatisticsForLegs(
            "", 'stats')



        predDfsDT, accsDT, legs, sumActsDT, sumPredsDT, percDiffsDT, stwsDT, drftsDT, latlonDep, latlonArr, sovgs = self.extractStatisticsForLegs(
            "NeuralDT", 'stats', )
        # self.extractStatisticsForLegs("NeuralDT")
        print("LEN legs :   " + str(len(legs)))

        ##########################  ======>
        totalSailingTimeHours, distTravelled, \
        depArrTime, instances, maxBalSpeed, maxLadSpeed, minDraft, maxDraft = self.extractInfoForPlots()

        ####plots
        # predDfsPL, accsPL, legsPL, sumActsPL, sumPredsPL, percDiffsPL, stwsPL, drftsPL, latlonDepPL, latlonArrPL = self.extractStatisticsForLegs(
        # "", 'plots')

        # predDfsDTPL, accsDTPL, legsPL, sumActsDTPL, sumPredsDTPL, percDiffsDTPL, stwsDTPL, drftsDTPL, latlonDepPL, latlonArrPL = self.extractStatisticsForLegs(
        # "NeuralDT", 'plots')

        self.extractPlotsForLegs(predDfs, predDfsDT, legs, accs, accsDT)
        #################

        predDfsOv, accsOV, legsOV, predDfsOvGW = self.extractOverallAccData("", 'stats',)

        predDfsOvDT, accsOVDT, legsOVDT, predDfsOvGWDT =  self.extractOverallAccData("NeuralDT", 'stats')

        ###Plots
        predDfsOvPL, accsOVPL, legsOVPL, predDfsOvGWPL = self.extractOverallAccData("", 'plots')

        #predDfsOvDTPL, accsOVDTPL, legsOVDTPL, predDfsOvGWDTPL = None, None, None, None
        # self.extractOverallAccData("NeuralDT", 'plots')

        self.plotOverallAcc(predDfsOv, accsOV, predDfsOvDT, accsOVDT, legsOV, legsOVDT, predDfsOvGW)
        ####END OF PLOTS

        predDfsWS, accsWS, legsWS, sumActsWS, sumPredsWS, percDiffsWS, stwsWS, drftsWS, latlonDepWS, latlonArrWS, sovgsWS = \
            self.extractStatisticsForLegs("NeuralDT", 'stats', True)

        templateRowTable = ''
        templateRowTableNDT = ''
        templateLegInfo = ''
        print(legs)

        ###### TABLES & LEGS INFO
        for i in range(0, len(legs)):


            ######################################################### NN TABLE INFO
            templateRowTableArgs = {'leg': legs[i], 'totalActFoc': "{:,.2f}".format(sumActs[i]),
                                    'totalPredFoc': "{:,.2f}".format(sumPreds[i]),
                                    'percDiff': percDiffs[i], 'meanSTW': stws[i], 'meanDraft': drfts[i], 'perc': '\%'}
            rowT = self.templateRowTable % templateRowTableArgs

            templateRowTable += rowT

            ######################################################### NDT TABLE INFO
            templateRowTableNDTArgs = {'leg': legs[i], 'totalActFoc': "{:,.2f}".format(sumActsDT[i]),
                                       'totalPredFoc': "{:,.2f}".format(sumPredsDT[i]),
                                       'percDiff': percDiffsDT[i], 'meanSTW': stwsDT[i], 'meanDraft': drftsDT[i],
                                       'perc': '\%'}
            rowTDT = self.templateRowTable % templateRowTableNDTArgs

            templateRowTableNDT += rowTDT

            ######################################################### LEGS INFO
            depDate = dataListLegs[i][0, 8]
            arrDate = dataListLegs[i][len(dataListLegs[i]) - 1, 8]
            rawLegs = preLegs.mapRawDataWithLegs(self.company, self.vessel, depDate, arrDate)
            self.extractInfoForWeatherComp(rawLegs, dataListLegsCorr[i], i)

            ####check if voyage is >= 3 days
            elapsedTime = (datetime.strptime(arrDate, "%Y-%m-%d %H:%M") - datetime.strptime(depDate, "%Y-%m-%d %H:%M"))
            totalSailingHoursRO = np.round(divmod(elapsedTime.total_seconds(), 60)[0] / 60, 2)
            #if totalSailingHoursRO <= 65 : #legs[i] != 'SINGAPORE - VUNG TAU'
            #if  legs[i] != 'SHEKOU - NINGBO':  # legs[i] != 'SINGAPORE - VUNG TAU'
                #continue

            ####check if voyage is >= 3 days

            ################# ROUTING OPTIMIZATION INFO ##############################################################
            ################# ROUTING OPTIMIZATION INFO ##############################################################

            path = './correctedLegsForTR/' + self.vessel + '/'
            templateForLegOpt = ''
            listOfFiles = self.sortLegsInFiles(path)

            countLegs = i

            fileLeg = listOfFiles[countLegs]
            leg = fileLeg.split("/")[3]
            # if leg != 'PUSAN_PANAMA_leg.csv': continue
            print("\n\nget STATS of opt for leg " + leg + ". . .")
            retas = [True, False]

            totalSailingTimeOpts = []
            stwOpts = []
            sumPredsOptFocs = []
            totalDistOpts = []
            retCodes = []
            naTrue = False
            naFalse = False

            for reta in retas:

                if os.path.isfile('./routingResponses/' + self.vessel + '/mappedResponse_' + self.vessel + '_' + str(
                        reta) + '_' + leg):
                    retCode = 1
                    fromAPI = False

                    ##change evalRoute and FromAPI boolean value to swithc between models
                else:
                    # retCode = -1
                    retCode = self.getOptimizedRouteFromNavigatorForLeg(leg, reta)
                    fromAPI = True
                # if os.path.isfile('./routingResponses/'+self.vessel+'/mappedResponse_'+self.vessel+'_'+leg) else -1
                # self.getOptimizedRouteFromNavigatorForLeg(leg)
                retCodes.append(retCode)
                if retCode == -1:
                    totalSailingTimeOpts.append("NA")
                    stwOpts.append("NA")
                    sumPredsOptFocs.append("NA")
                    totalDistOpts.append("NA")
                    self.plotOptimizationVSInitialWeather(leg, countLegs + 1, reta, notAvailable=True)
                    continue

                if reta == True:

                    #print("Mapping response with WS . . .")
                    #self.mapResponseRoute(leg, 'optimized', reta)


                    print("Evaluating routes from Neural DT . . . ")
                    self.evalRoutes(leg, "optimized", reta)

                    self.plotOptimizationVSInitialWeather(leg, countLegs + 1, reta, )

                    self.saveResponseRouteCoords(leg, reta)
                    #if leg =='PUSAN_PANAMA_leg.csv':
                    #self.saveResponseRouteCoords(leg, reta)
                    self.drawRoutesOnMap(leg, "response", reta)

                    totalSailingTimeOpt, stwOpt, sumPredsOptFoc, totalDistOpt = self.getStatsOfOPtimization(leg, fromAPI, reta)
                    ###append True False RestrictETA
                    totalSailingTimeOpts.append(totalSailingTimeOpt)
                    stwOpts.append(stwOpt)
                    sumPredsOptFocs.append(sumPredsOptFoc)
                    totalDistOpts.append(totalDistOpt)

            if np.sum(retCodes) == -2: continue

            opt_init_templateTrue = r"\textbf{NA}" if retCodes[0] == -1 else self.opt_init_templateTrue
            opt_init_templateFalse = r"\textbf{NA}" if retCodes[1] == -1 else self.opt_init_templateFalse

            naTrue = True if retCodes[0] == -1 else naTrue
            naFalse = True if retCodes[1] == -1 else naFalse

            legWOExt = leg.split(".")[0]
            if str(leg).__contains__(" "):
                leg1 = legWOExt.split(" ")[0]
                leg2 = legWOExt.split(" ")[1]
            legF = leg1 + "_" + leg2 if str(leg).__contains__(" ") else legWOExt

            print("Map LEG: " +legF)
            '''if leg == 'SUEZ_SINGAPORE_leg.csv':
                totalFocOptTrue = 909.37

            elif leg == 'PUSAN_PANAMA_leg.csv':
                totalFocOptTrue = 2221.71
            else:
                totalFocOptTrue = 670.92'''

            templateROLegInfoArgs = {'leg': legs[countLegs],
                                     'company': self.company,
                                     'vessel': self.vslFig,
                                     'legNumber': countLegs + 1,
                                     'legFig': legF,
                                     'depDate': depArrTime[countLegs].split(" - ")[0],
                                     'arrDate': depArrTime[countLegs].split(" - ")[1],
                                     'latDep': latlonDep[countLegs].split("-")[0],
                                     'lonDep': latlonDep[countLegs].split("-")[1],
                                     'latArr': latlonArr[countLegs].split("-")[0],
                                     'lonArr': latlonArr[countLegs].split('-')[1],
                                     'totalFocOptTrue': "{:,.2f}".format(sumPredsOptFocs[0]) if sumPredsOptFocs[0] != "NA" else "NA",
                                     #'totalFocOptTrue': "{:,.2f}".format(totalFocOptTrue),

                                     'totalFocEst': "{:,.2f}".format(sumPredsWS[countLegs]),
                                     # 'totalFocOptFalse': "{:,.2f}".format(sumPredsOptFocs[1])  if sumPredsOptFocs[1]!="NA" else sumPredsOptFocs[1],

                                     'distNMOptTrue': "{:,.2f}".format(totalDistOpts[0]) if totalDistOpts[
                                                                                                0] != "NA" else
                                     totalDistOpts[0],
                                     'distNMEst': "{:,.2f}".format(sovgs[countLegs] * totalSailingTimeHours[countLegs]),
                                     # 'distNMOptFalse': "{:,.2f}".format(totalDistOpts[1]) if totalDistOpts[1]!="NA" else totalDistOpts[1],

                                     'avgSpeedOptTrue': "{:,.2f}".format(np.mean(stwOpts[0])) if stwOpts[0] != "NA" else
                                     stwOpts[0],
                                     'avgSpeed': "{:,.2f}".format(sovgs[countLegs]),
                                     # 'avgSpeedOptFalse': "{:,.2f}".format(np.mean(stwOpts[1])) if stwOpts[1]!="NA" else stwOpts[1],

                                     'totalTimeOptTrue': "{:,.2f}".format(totalSailingTimeOpts[0]) if
                                     totalSailingTimeOpts[0] != "NA" else totalSailingTimeOpts[0],
                                     'totalTime': "{:,.2f}".format(totalSailingTimeHours[countLegs]),
                                     # 'totalTimeOptFalse': "{:,.2f}".format(totalSailingTimeOpts[1])  if totalSailingTimeOpts[1]!="NA" else totalSailingTimeOpts[1],

                                     "scaleTrue": '0.12' if naTrue == True else "",
                                     "scaleFalse": '0.12' if naFalse == True else "",
                                     'perc': "%"

                                     }

            rowT = self.templateROLegInfo % templateROLegInfoArgs
            # templateForLegOpt += rowT
            #### END RO ###############################
            #### END RO ###############################
            #### END RO ###############################

            templateLegInfoArgs = {'leg': legs[i], 'legNumber': i + 1, 'legNumberFig': i,
                                   'totalActFoc': "{:,.2f}".format(sumActs[i]),
                                   'totalPredFoc': "{:,.2f}".format(sumPreds[i]),
                                   'totalPredFocDT': "{:,.2f}".format(sumPredsDT[i]),
                                   'percDiff': percDiffs[i],
                                   'percDiffDT': percDiffsDT[i],
                                   'depPort': legs[i].split("-")[0], 'depDate': depArrTime[i].split(" - ")[0],
                                   'arrPort': legs[i].split("-")[1], 'arrDate': depArrTime[i].split(" - ")[1],
                                   'totalTime': "{:,.0f}".format(totalSailingTimeHours[i]),
                                   'totalDist': "{:,.2f}".format(distTravelled[i]), 'meanDraft': drfts[i],
                                   'company': self.company, 'vessel': self.vslFig, 'perc': '\%',
                                   'templateROLegInfo': rowT
                                   }

            legInf = self.templateLegInfo % templateLegInfoArgs

            templateLegInfo += legInf

        # self.templateRowTable =  templateRowTable
        self.templateLegInfo = templateLegInfo

        totalFocAct = np.sum(sumPreds)
        totalFocPred = np.sum(sumActs)

        nom = abs(totalFocAct - totalFocPred)
        denom = (totalFocAct + totalFocPred) / 2
        percDiff = nom / denom
        percDiff *= 100

        ### percDiff Neural DT
        totalFocActDT = np.sum(sumPredsDT)
        totalFocPredDT = np.sum(sumActsDT)

        nom = abs(totalFocActDT - totalFocPredDT)
        denom = (totalFocActDT + totalFocPredDT) / 2
        percDiffDT = nom / denom
        percDiffDT *= 100

        ######################################### Neural Net #############################################
        templateSummaryTableArgs = {'templateRowTable': templateRowTable, 'vslFig': self.vslFig, 'vessel': self.vessel,
                                    'model': 'Neural Net',
                                    'totalAvgActFoc': "{:,.2f}".format(np.sum(sumActs)),
                                    'totalAvgPredFoc': "{:,.2f}".format(np.sum(sumPreds)),
                                    'totalAvgPercDiff': np.round(percDiff, 2),
                                    'totalAvgSTW': np.mean(stws),
                                    'totalAvgDraft': np.mean(drfts),
                                    'totalInstances': "{:,.0f}".format(instances),
                                    'totalTime': "{:,.0f}".format(np.sum(totalSailingTimeHours)),
                                    'totalDist': "{:,.2f}".format(np.sum(distTravelled)), 'company': self.company,
                                    'perc': '\%'}

        templateSummaryTableNN = self.templateSummaryTable % templateSummaryTableArgs

        #########################################> Neural DT <#############################################
        templateSummaryTableArgsNDT = {'templateRowTable': templateRowTableNDT, 'vslFig': self.vslFig,
                                       'model': 'Neural DT',
                                       'vessel': self.vessel,
                                       'totalAvgActFoc': "{:,.2f}".format(np.sum(sumActsDT)),
                                       'totalAvgPredFoc': "{:,.2f}".format(np.sum(sumPredsDT)),
                                       'totalAvgPercDiff': np.round(percDiffDT, 2),
                                       'totalAvgSTW': np.mean(stwsDT),
                                       'totalAvgDraft': np.mean(drftsDT),
                                       'totalInstances': "{:,.0f}".format(instances),
                                       'totalTime': "{:,.0f}".format(np.sum(totalSailingTimeHours)),
                                       'totalDist': "{:,.2f}".format(np.sum(distTravelled)), 'company': self.company,
                                       'perc': '\%'}

        templateSummaryTableNDT = self.templateSummaryTable % templateSummaryTableArgsNDT

        # ====================> <=================================

        # overall accuracy Plot and Info
        templateOvAccArgs = {'totalInstances': "{:,.0f}".format(instances),
                             'totalTime': "{:,.0f}".format(np.sum(totalSailingTimeHours)),
                             'totalDist': "{:,.2f}".format(np.sum(distTravelled)), 'company': self.company,
                             'vslFig': self.vslFig,
                             }

        templateOvAcc = self.overallAcc % templateOvAccArgs

        ################################### optimization INFO

        ###Weather comp Template
        data = pd.read_csv('./data/' + self.company + '/' + self.vessel + '/mappedData.csv').values
        dataCorr = pd.read_csv('./consProfileJSON_Neural/cleaned_raw_' + self.vessel + '.csv').values
        # preLegs.concatLegs(self.vessel, './correctedLegsForTR/'+self.vessel+'/', False)
        self.extractInfoForWeatherComp(data, dataCorr)
        weatherCompTemplateArgs = {
            'company': self.company,
            'vessel': self.vslFig,
        }

        weatherCompTemplate = self.weatherCompTemplate % weatherCompTemplateArgs

        ###main args ####
        args = {'vessel': orVslName, 'imo': str(imo), 'vesselType': vslType, 'minDraft': minDraft, 'maxDraft': maxDraft,
                'templateSummaryTableNNet': templateSummaryTableNN, 'templateSummaryTableNDT': templateSummaryTableNDT,
                'templateOvAcc': templateOvAcc, 'templateLegInfo': self.templateLegInfo,
                'maxBalSpeed': maxBalSpeed, 'maxLadSpeed': maxLadSpeed, 'weatherCompTemplate': weatherCompTemplate}

        with open(self.vessel + '.tex', 'w') as f:
            f.write(self.main % args)

        cmd = ['pdflatex', '-interaction', 'nonstopmode', self.vessel + '.tex']
        proc = subprocess.Popen(cmd)
        proc.communicate()

        retcode = proc.returncode
        if not retcode == 0:
            os.unlink('./' + self.vessel + '.pdf')
            raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))

        os.unlink(self.vessel + '.tex')
        os.unlink(self.vessel + '.log')

    def extractOverallAccData(self, model, type):

        rawGW = []
        #if model == "":
        path = './sensAnalysis' + model + '/' + self.vessel + '/'
        #else:
            #path = './sensAnalysis' + model + '/' + self.vessel + '/'

        predDfs = []
        predDfsGW = []
        legs = []
        accs = []
        dataSet = []
        legs.append('Overall Accuracy on Legs')
        for infile in sorted(glob.glob(path + '*.csv')):
            leg = infile.split('/')[3]
            # if leg == 'preds_GENOA_LONDON_ANTWERP_leg.csv': continue
            print(leg)
            testPred = pd.read_csv(infile, sep=',', decimal='.')
            #if type == 'plots':
            testPred = testPred[testPred['stw'] >= 12]
            testPred = testPred.reset_index(drop=True)
            testPred = testPred.values
            # print(testPred['FOC_act'])
            dataSet.append(testPred)

        dataSet = np.concatenate(dataSet)

        # stwMaxBallast = np.max(np.array([k for k in dataSet if k[1] >= 9 and k[1] <= 11])[:, 0])
        # stwMaxLaden = np.max(np.array([k for k in dataSet if k[1] > 9 and k[1] <= 14])[:, 0])
        # print(stwMaxBallast)
        # print(stwMaxLaden)

        print(dataSet.shape)
        testPred = dataSet
        # print(testPred)

        ##calc FOC/MILE
        focsPerMile = []
        focsPerMileGW = []
        focsPerMilePred = []
        stwGW = []

        for i in range(0, len(testPred)):

            stw = testPred[i, 0] if model == 0 else testPred[i, 3]
            if i == 0:
                stwPrevKMperMin = (stw) / 32.39741
                distKM = stwPrevKMperMin * 1
                distMiles = distKM / 1.609

            if i > 0:
                dt = datetime.strptime(testPred[i, 7] if model == "" else testPred[i, 5], "%Y-%m-%d %H:%M:%S")
                dtPrev = datetime.strptime(testPred[i - 1, 7] if model == "" else testPred[i, 5], "%Y-%m-%d %H:%M:%S")

                elapsedTimeFromPrev = (dt - dtPrev)
                minsFromPrev = np.round(divmod(elapsedTimeFromPrev.total_seconds(), 60)[0], 2)
                stwPrevKMperMin = (stw) / 32.39741
                distKM = stwPrevKMperMin * minsFromPrev
                distMiles = distKM / 1.609

            focPerMile = (testPred[i, 5] / 1000) / distMiles if model == "" else (testPred[i, 1] / 1440) / distMiles
            focPerMilePred = (testPred[i, 6] / 1000) / distMiles if model == "" else (testPred[i, 2] / 1440) / distMiles
            if np.isinf(focPerMile):
                focsPerMile.append(focsPerMile[i - 1])
                if testPred[i, 2] >= 0 and testPred[i, 2] <= 3:
                    focsPerMileGW.append(focsPerMile[i - 1])
                    stwGW.append(testPred[i, 0])
                focsPerMilePred.append(focsPerMilePred[i - 1])
            else:
                focsPerMile.append(focPerMile)
                if testPred[i, 2] >= 0 and testPred[i, 2] <= 3:
                    focsPerMileGW.append(focPerMile)
                    stwGW.append(testPred[i, 0])
                focsPerMilePred.append(focPerMilePred)

        focsPerMile = np.array(focsPerMile)
        focsPerMilePred = np.array(focsPerMilePred)
        stwGW = np.array(stwGW)

        ###end foc per mile ####################################

        if model == "":
            # stw,draft,ws,wd,swh,FOC_act,FOC_pred
            testPred[:, 6] = (testPred[:, 6] / 1000) * 1440
            testPred[:, 5] = (testPred[:, 5] / 1000) * 1440

            meanAcc = ((np.abs(testPred[:, 6] - testPred[:, 5])) / testPred[:, 5]) * 100
            mae = np.abs(testPred[:, 5] - testPred[:, 6])

            data = testPred
            raw = np.array(np.append(data[:, 0].reshape(-1, 1),
                                     np.asmatrix([data[:, 1], data[:, 5].astype(float), data[:, 6], focsPerMile,
                                                  focsPerMilePred]).T, axis=1))

            rawGW = np.array(np.append(stwGW.reshape(-1, 1),
                                       np.asmatrix([focsPerMileGW, ]).T, axis=1))


        else:

            meanAcc = ((np.abs(testPred[:, 2] - testPred[:, 1])) / testPred[:, 1]) * 100
            mae = np.abs(testPred[:, 1] - testPred[:, 2])

            data = testPred
            raw = np.array(np.append(data[:, 3].reshape(-1, 1),
                                     np.asmatrix([data[:, 4], data[:, 2].astype(float), data[:, 1], focsPerMile,
                                                  focsPerMilePred]).T, axis=1))

        meanAcc = str(np.round(100 - np.mean(meanAcc), 2)) + '%'
        print("MP Accuracy " + str(meanAcc))

        print("MAE " + str(np.round(np.mean(mae))))

        # print(raw)
        minSpeed = np.min(raw[:, 0])
        i = minSpeed
        maxSpeed = np.max(raw[:, 0])
        sizesSpeed = []
        speed = []
        speedGW = []
        avgActualFoc = []
        minActualFoc = []
        maxActualFoc = []
        stdActualFoc = []
        avgPredFoc = []
        draft = []
        meanFocsPerMileGW = []
        meanFocsPerMilePred = []
        meanFocsPerMile = []
        while i <= maxSpeed:
            # workbook._sheets[sheet].insert_rows(k+27)

            speedArray = np.array([k for k in raw if float(k[0]) >= i - 0.25 and float(k[0]) <= i + 0.25])

            speedArrayGW = np.array([k for k in rawGW if float(k[0]) >= i - 0.25 and float(k[0]) <= i + 0.25])

            if len(speedArray) > 0:
                sizesSpeed.append(len(speedArray))
                speed.append(i)
                draft.append(np.mean(speedArray[:, 1]))
                avgActualFoc.append(np.mean(speedArray[:, 2]) if len(speedArray) > 0 else 0)
                avgPredFoc.append(np.mean(speedArray[:, 3]) if len(speedArray) > 0 else 0)
                minActualFoc.append(np.min(speedArray[:, 2]) if len(speedArray) > 0 else 0)
                maxActualFoc.append(np.max(speedArray[:, 2]) if len(speedArray) > 0 else 0)
                stdActualFoc.append(np.std(speedArray[:, 2]) if len(speedArray) > 0 else 0)
                meanFocsPerMile.append(np.mean(speedArray[:, 4]))
                meanFocsPerMilePred.append(np.mean(speedArray[:, 5]))

            if len(speedArrayGW) > 0:
                speedGW.append(i)
                meanFocsPerMileGW.append(np.mean(speedArrayGW[:, 1]))

            i += 0.5

        plt.plot(speed, np.abs(np.array(avgActualFoc) - np.array(avgPredFoc)))
        plt.xlabel('speed')
        plt.ylabel('MAE')
        plt.grid()
        # plt.show()

        predDf_ = pd.DataFrame(
            {'speed': np.round(speed, 2), 'draft': draft, 'pred_Foc': avgPredFoc, 'actual_Foc': avgActualFoc,
             'size': sizesSpeed, 'focPerMile': meanFocsPerMile, 'focPerMilePred': meanFocsPerMilePred})
        predDf_['speed'] = [str(k) for k in predDf_['speed'].values]
        predDf_
        predDfs.append(predDf_)

        ###GW
        predDfGW_ = pd.DataFrame(
            {'speed': np.round(speedGW, 2), 'focPerMileGW': meanFocsPerMileGW})
        predDfGW_['speed'] = [str(k) for k in predDfGW_['speed'].values]
        predDfGW_
        predDfsGW.append(predDfGW_)

        accs.append(meanAcc)

        return predDfs, accs, legs, predDfsGW

    def plotOverallAcc(self, predDfs, accs, predfsDT, accsDT, legs, legsDT, predDfsGW):

        plt.clf()
        for predDf_, predDf_DT, leg, legDT, meanAcc, meanAccDT, predDfGW_ in zip(predDfs, predfsDT, legs, legsDT, accs,
                                                                                 accsDT, predDfsGW):
            df = predDf_
            dfDT = predDf_DT
            dfGW = predDfGW_

            df['speed'] = [str(k) for k in df['speed'].values]
            dfDT['speed'] = [str(k) for k in dfDT['speed'].values]

            fig, ax1 = plt.subplots(figsize=(15, 10))

            ax1 = sns.barplot(x='speed', y='size', data=df, palette='summer', ci="sd")
            ax1.set_xlabel('Speed Ranges (knots)', fontsize=16)
            ax1.set_ylabel('Size', fontsize=16)
            # change_width(ax1, 1.5)
            ax1.tick_params(axis='x')
            ax1.set_xticks(np.arange(9, 15, step=15.2))
            ax1.grid()
            # specify we want to share the same x-axis
            ax2 = ax1.twinx()

            #if legDT != "": ax2 = sns.lineplot(x='speed', y='pred_Foc', data=dfDT, sort=False, color='green',
                                               #label='neuralDT Foc Acc: ' + meanAccDT)
            ax2 = sns.lineplot(x='speed', y='pred_Foc', data=df, sort=False, color='blue',
                               label='neural Foc Acc: ' + meanAcc)
            ax2 = sns.lineplot(x='speed', y='actual_Foc', data=df, sort=False, color='red', label='actual Foc')
            ax2.set_ylabel('Foc (MT/day)', fontsize=16)
            ax2.grid()
            plt.title(leg)

            plt.legend()
            plt.grid()
            plt.savefig('./Figures/' + self.company + '/' + self.vslFig + '/overallAcc.eps', format='eps')

            ###per mile

            fig, ax1 = plt.subplots(figsize=(15, 10))

            ax1 = sns.barplot(x='speed', y='size', data=df, palette='summer', ci="sd")
            ax1.set_xlabel('Speed Ranges (knots)', fontsize=16)
            ax1.set_ylabel('Size', fontsize=16)
            # change_width(ax1, 1.5)
            ax1.tick_params(axis='x')
            ax1.set_xticks(np.arange(9, 15, step=15.2))
            ax1.grid()
            # specify we want to share the same x-axis
            ax2 = ax1.twinx()

            # if legDT != "" : ax2 = sns.lineplot(x='speed', y='focPerMilePred', data=dfDT, sort=False, color='green',label='neuralDT Foc / Mile Acc: ' + meanAccDT)

            ax2 = sns.lineplot(x='speed', y='focPerMilePred', data=df, sort=False, color='blue',
                               label='neural Foc / Mile Acc: ' + meanAcc)
            ax2 = sns.lineplot(x='speed', y='focPerMile', data=df, sort=False, color='red', label='actual Foc / mile')
            ax2 = sns.lineplot(x='speed', y='focPerMileGW', data=dfGW, sort=False, color='green',
                               label='actual Foc / mile [0 - 3 Bft]')
            ax2.set_ylabel('Foc (MT/day)', fontsize=16)
            ax2.grid()
            plt.title(leg)

            plt.legend()
            plt.grid()
            plt.savefig('./Figures/' + self.company + '/' + self.vslFig + '/overallAccPerMile.eps', format='eps')
            # plt.show()

    def extractPlotsForLegs(self, predDfs, predDfsDT, legs, accs, accsDT):

        from sklearn.metrics import mean_absolute_error
        # from sklearn.metrics import mean_absolute_percentage_error
        for i, predDf_, predDf_DT, leg, meanAcc, meanAccDT in zip(range(0, len(predDfs)), predDfs, predDfsDT, legs,
                                                                  accs, accsDT):
            df = predDf_
            df['speed'] = [str(k) for k in df['speed'].values]

            dfDT = predDf_DT
            dfDT['speed'] = [str(k) for k in dfDT['speed'].values]
            fig, ax1 = plt.subplots(figsize=(15, 10))
            color = 'tab:green'
            # bar plot creation
            y_pos = [0, 1, 5, 8, 9, 11, 12, 13, 14]
            bars = df['size']
            # plt.xticks(y_pos, bars)

            ax1 = sns.barplot(x='speed', y='size', data=df, palette='summer', ci="sd")
            ax1.set_xlabel('Speed Ranges (knots)', fontsize=16)
            ax1.set_ylabel('Size', fontsize=16)
            # change_width(ax1, 1.5)
            ax1.tick_params(axis='x')
            ax1.set_xticks(np.arange(9, 15, step=15.2))
            ax1.grid()
            # specify we want to share the same x-axis
            ax2 = ax1.twinx()
            color = 'tab:red'

            #ax2 = sns.lineplot(x='speed', y='pred_Foc', data=dfDT, sort=False, color='green',
                               #label='neuralDT Foc Acc: ' + meanAccDT)
            ax2 = sns.lineplot(x='speed', y='pred_Foc', data=df, sort=False, color='blue',
                               label='neural Foc Acc: ' + meanAcc)
            ax2 = sns.lineplot(x='speed', y='actual_Foc', data=df, sort=False, color='red', label='actual Foc')
            ax2.set_ylabel('Foc (MT/day)', fontsize=16)
            ax2.grid()
            plt.title(leg)

            plt.legend()
            plt.grid()
            plt.savefig('./Figures/' + self.company + '/' + self.vslFig + '/leg' + str(i) + '.eps', format='eps')

            # plt.show()

    def decdeg2dms(self, dd):

        mnt, sec = divmod(dd * 3600, 60)

        deg, mnt = divmod(mnt, 60)

        return abs(deg), abs(mnt), abs(sec)

    def extractStatisticsForLegs(self, model, type, ws = None):

        if model == "" and ws ==True:
            path = './sensAnalysisWS' + model + '/' + self.vessel + '/'
        elif model == "" and ws == False:
            path = './sensAnalysis' + model + '/' + self.vessel + '/'
        else:
            path = './sensAnalysis' + model + '/' + self.vessel + '/'

        sorted_legs = self.sortLegsInFiles(path)
        predDfs = []
        legs = []
        accs = []
        sumPreds = []
        sumActs = []
        stws = []
        drfts = []
        percDiffs = []
        latlonDep = []
        latlonArr = []

        # sorted(glob.glob(path + '*.csv'))
        for infile in sorted_legs:

            legInit = infile.split('/')[3]
            legInitWithoutVN = legInit.split(self.vessel+"_")[1]
            leg = legInit.split("_")[2] + " - " + legInit.split("_")[3]



            testPred = pd.read_csv(infile, sep=',', decimal='.')
            if type == 'plots':
                testPred = testPred[testPred['stw'] >= 12]
                testPred = testPred.reset_index(drop=True)
            '''if len(testPred) < 1000:
                os.remove(path+'/'+legInit);
                if os.path.isdir('./legs/' + legInitWithoutVN): os.remove('./legs/'+legInitWithoutVN);
                if os.path.isdir('./correctedLegsForTR/'+legInitWithoutVN): os.remove('./correctedLegsForTR/'+legInitWithoutVN);
                continue'''
            #if len(testPred) < 1000: continue
            legs.append(leg)
            #print(leg)
            ##lat lon dep arr
            legFile = infile.split(self.vessel + "_")[1]
            legDat = pd.read_csv('./legs/' + self.vessel + '/' + legFile).values
            latDep = legDat[0][6]
            lonDep = legDat[0][7]

            latArr = legDat[len(legDat) - 1][6]
            lonArr = legDat[len(legDat) - 1][7]

            latDepDMS = self.decdeg2dms(latDep)
            lonDepDMS = self.decdeg2dms(lonDep)

            latDir = 'S' if latDep < 0 else 'N'
            lonDir = 'W' if latDep < 0 else 'E'

            latDepDM = str(latDepDMS[0]) + u"\N{DEGREE SIGN}" + "  " + str(latDepDMS[1]) + "'" + "  " + str(latDir)
            lonDepDM = str(lonDepDMS[0]) + u"\N{DEGREE SIGN}" + "  " + str(lonDepDMS[1]) + "'" + "  " + str(lonDir)

            #####################
            latArrDMS = self.decdeg2dms(latArr)
            lonArrDMS = self.decdeg2dms(lonArr)

            latDir = 'N' if latArr < 0 else 'S'
            lonDir = 'W' if latArr < 0 else 'E'

            latArrDM = str(latArrDMS[0]) + u"\N{DEGREE SIGN}" + "  " + str(latArrDMS[1]) + "'" + "  " + str(latDir)
            lonArrDM = str(lonArrDMS[0]) + u"\N{DEGREE SIGN}" + "  " + str(lonArrDMS[1]) + "'" + "  " + str(lonDir)

            # if i ==0 :
            latlonDep.append(latDepDM + "-" + lonDepDM)
            latlonArr.append(latArrDM + "-" + lonArrDM)
            ##lat lon dep arr

            print(testPred.shape)

            if model == "":
                # lt/min => MT/day
                testPredFOC_pred = np.round((testPred['FOC_pred'] / 1000) * 1440, 2)
                testPredFOC_act = np.round((testPred['FOC_act'] / 1000) * 1440, 2)

            if model == "":
                focPreds = []
                focActs = []
                for i in range(0, len(testPred)):
                    if i > 0:
                        dt = datetime.strptime(testPred['dt'][i], "%Y-%m-%d %H:%M:%S")
                        dtPrev = datetime.strptime(testPred['dt'][i - 1], "%Y-%m-%d %H:%M:%S")
                        elapsedTimeFromPrev = (dt - dtPrev)
                        minsFromPrev = np.round(divmod(elapsedTimeFromPrev.total_seconds(), 60)[0], 2)
                        focPreds.append((testPred['FOC_pred'][i] / 1000) * minsFromPrev)
                        focActs.append((testPred['FOC_act'][i] / 1000) * minsFromPrev)
                    else:

                        focPreds.append((testPred['FOC_pred'][i] / 1000))
                        focActs.append((testPred['FOC_act'][i] / 1000))

                # sumPred = np.round(np.sum(testPred['FOC_pred'].values/1000),2)
                # sumAct = np.round(np.sum(testPred['FOC_act'].values/1000),2)
                sumPred = np.round(np.sum(focPreds), 2)
                sumAct = np.round(np.sum(focActs), 2)



            else:
                ###neural DT

                focPreds = []
                focActs = []
                for i in range(0, len(testPred)):
                    if i > 0:
                        dt = datetime.strptime(testPred['dt'][i], "%Y-%m-%d %H:%M:%S")
                        dtPrev = datetime.strptime(testPred['dt'][i - 1], "%Y-%m-%d %H:%M:%S")
                        elapsedTimeFromPrev = (dt - dtPrev)
                        minsFromPrev = np.round(divmod(elapsedTimeFromPrev.total_seconds(), 60)[0], 2)
                        focPreds.append((testPred['FOC_pred'][i] / 1440) * minsFromPrev)
                        focActs.append((testPred['FOC_act'][i] / 1440) * minsFromPrev)

                    else:

                        focPreds.append((testPred['FOC_pred'][i] / 1440))
                        focActs.append((testPred['FOC_act'][i] / 1440))

                # sumPred = np.round(np.sum(testPred['FOC_pred'].values/1000),2)
                # sumAct = np.round(np.sum(testPred['FOC_act'].values/1000),2)
                sumPred = np.round(np.sum(focPreds), 2)
                sumAct = np.round(np.sum(focActs), 2)

                # sumPred = np.round(np.sum(testPred['FOC_pred'].values / 1440), 2)
                # sumAct = np.round(np.sum(testPred['FOC_act'].values / 1440), 2)

            sumPreds.append(sumPred)
            sumActs.append(sumAct)

            nom = abs(sumPred - sumAct)
            denom = (sumPred + sumAct) / 2
            percDiff = nom / denom
            percDiff *= 100
            percDiffs.append(np.round(percDiff, 2))
            # print(leg + " Sum Act " + str(sumAct))
            # print(leg + " Sum Pred " + str(sumPred))

            # print("Perc diff: " + str(((abs(sumPred - sumAct) / (sumPred + sumAct) / 2)) * 100))
            # print("Mean STW: " + str(np.mean(testPred['stw'])))
            # print("Mean Draft: " + str(np.mean(testPred['draft'])))

            stws.append(np.mean(testPred['stw']))
            drfts.append(np.mean(testPred['draft']))

            if model == "": testPred['FOC_pred'] = testPredFOC_pred
            if model == "": testPred['FOC_act'] = testPredFOC_act

            meanAcc = ((np.abs(testPred['FOC_pred'] - testPred['FOC_act'])) / testPred['FOC_act']) * 100

            meanAcc = str(np.round(100 - np.mean(meanAcc), 2)) + '%'
            #print("MP Accuracy " + str(meanAcc))

            mae = np.abs(testPred['FOC_act'] - testPred['FOC_pred'])
            #print("MAE " + str(np.round(np.mean(mae))))

            data = testPred.values

            if model == "":
                raw = np.array(np.append(data[:, 0].reshape(-1, 1),
                                         np.asmatrix([data[:, 1], data[:, 5].astype(float), data[:, 6]]).T, axis=1))
            else:
                raw = np.array(np.append(data[:, 3].reshape(-1, 1),
                                         np.asmatrix([data[:, 4], data[:, 2].astype(float), data[:, 1]]).T, axis=1))
            # print(raw)
            #minSpeed = np.min(raw[:, 0])
            minSpeed = 12
            i = minSpeed
            maxSpeed = np.max(raw[:, 0])
            sizesSpeed = []
            speed = []
            avgActualFoc = []
            minActualFoc = []
            maxActualFoc = []
            stdActualFoc = []
            avgPredFoc = []
            draft = []
            while i <= maxSpeed:
                # workbook._sheets[sheet].insert_rows(k+27)

                speedArray = np.array([k for k in raw if float(k[0]) >= i - 0.25 and float(k[0]) <= i + 0.25])

                if len(speedArray) > 0:
                    sizesSpeed.append(len(speedArray))
                    speed.append(i)
                    draft.append(np.mean(speedArray[:, 1]))
                    avgActualFoc.append(np.mean(speedArray[:, 2]) if len(speedArray) > 0 else 0)
                    avgPredFoc.append(np.mean(speedArray[:, 3]) if len(speedArray) > 0 else 0)
                    minActualFoc.append(np.min(speedArray[:, 2]) if len(speedArray) > 0 else 0)
                    maxActualFoc.append(np.max(speedArray[:, 2]) if len(speedArray) > 0 else 0)
                    stdActualFoc.append(np.std(speedArray[:, 2]) if len(speedArray) > 0 else 0)
                i += 0.5
            plt.plot(speed, np.abs(np.array(avgActualFoc) - np.array(avgPredFoc)))
            plt.xlabel('speed')
            plt.ylabel('MAE')
            plt.grid()
            # plt.show()

            predDf_ = pd.DataFrame(
                {'speed': np.round(speed, 2), 'draft': draft, 'pred_Foc': avgPredFoc, 'actual_Foc': avgActualFoc,
                 'size': sizesSpeed})
            predDf_['speed'] = [str(k) for k in predDf_['speed'].values]

            predDfs.append(predDf_)
            accs.append(meanAcc)

        # ========> NeuralDT ###########################  <========

        # ========> NeuralDT ###########################  <========
        path = './correctedLegsForTR/' + self.vessel + '/'
        sorted_legs = self.sortLegsInFiles(path)
        sovgs = []
        for infile in sorted_legs:
            legInfo = pd.read_csv(infile, sep=',', decimal='.').values
            sovgs.append(np.mean(legInfo[:,11]))

        return predDfs, accs, legs, sumActs, sumPreds, percDiffs, stws, drfts, latlonDep, latlonArr, sovgs

    def extractInfoForPlots(self, ):
        ##extract info for Plots
        path = './legs/' + self.vessel + '/'
        # path = './correctedLegsForTR/' + self.vessel + '/'
        sorted_legs = self.sortLegsInFiles(path)
        totalSailingTimeHours = []
        distTravelled = []
        depArrTime = []
        instances = []
        data = []
        for infile in sorted_legs:
            legInfo = pd.read_csv(infile, sep=',', decimal='.')

            data.append(legInfo.values)
            instances.append(len(legInfo))
            originalDeptlgDate = datetime.strptime(legInfo['dt'][0], '%Y-%m-%d %H:%M')
            originalArrtlgDate = datetime.strptime(legInfo['dt'][len(legInfo) - 1], '%Y-%m-%d %H:%M')

            lat_lon_Dep = (legInfo['lat'][0], legInfo['lon'][0])
            lat_lon_Arr = (legInfo['lat'][len(legInfo) - 1], legInfo['lon'][len(legInfo) - 1])

            elapsedTime = (originalArrtlgDate - originalDeptlgDate)

            totalSailingTime = np.round(divmod(elapsedTime.total_seconds(), 60)[0] / 60, 2)
            totalDist = geopy.distance.distance(lat_lon_Dep, lat_lon_Arr).nm
            totalSailingTimeHours.append(totalSailingTime)
            distTravelled.append(totalDist)
            depArrTime.append(str(originalDeptlgDate) + ' - ' + str(originalArrtlgDate))

        data = np.array(np.concatenate(data))

        dataModel = KMeans(n_clusters=4)
        drafts = data[:, 1]
        drafts = drafts.reshape(-1, 1)
        dataModel.fit(drafts)
        # Extract centroid values
        centroids = dataModel.cluster_centers_
        draftsSorted = np.sort(centroids, axis=0)

        # minDraft = np.floor(min(data[:, 1]))
        # maxDraft = np.ceil(max(data[:, 1]))
        minDraft = np.floor(draftsSorted[0][0])
        maxDraft = np.ceil(draftsSorted[3][0])

        drafts = np.linspace(minDraft, maxDraft, 3)
        maxBalSpeed = np.max(np.array([k for k in data if k[1] >= drafts[0] and k[1] < drafts[1]])[:, 0])
        maxLadSpeed = np.max(np.array([k for k in data if k[1] >= drafts[1] and k[1] <= drafts[2]])[:, 0])
        #################################
        ### exctract Pplots for Legs

        return totalSailingTimeHours, distTravelled, \
               depArrTime, np.sum(instances), maxBalSpeed, maxLadSpeed, minDraft, maxDraft

    def getStatsOfOPtimization(self, leg, fromAPI, reta):

        if fromAPI == True:
            file = self.ndt.getCurrentGeneratedFileFromWebAPI(self.pathToWebAPIForPavlosInterpolation)
            print("API file: " + file + '\n\n')
            df = pd.read_csv(file)
            df.to_csv('./APIresp/' + self.vessel + '/compareSpeeddetailed_interp_' + str(reta) + '_' + leg)
            print("API generated file writen in: " + './APIresp/' + self.vessel + '/compareSpeeddetailed_interp_' + str(
                reta) + '_' + leg)
            os.remove(file)
        else:

            df = pd.read_csv('./APIresp/' + self.vessel + '/compareSpeeddetailed_interp_' + str(reta) + '_' + leg)

        focPredsOptFoc = df['FOC_pred'].values
        stwOpt = df['stw'].values

        focPredsOpt = []

        # sumPredsOptFoc  = np.sum(focPredsOptFoc)

        resp = pd.read_csv(
            './routingResponses/' + self.vessel + '/mappedResponse_' + self.vessel + '_' + str(reta) + '_' + leg).values
        dist = 0
        totalDist = 0
        for i in range(0, len(resp)):

            lat = resp[i][6]
            lon = resp[i][7]

            if i > 0:

                dt = datetime.strptime(resp[i][5].split(".")[0], "%Y-%m-%d %H:%M:%S")
                dtPrev = datetime.strptime(resp[i - 1][5].split(".")[0], "%Y-%m-%d %H:%M:%S")
                elapsedTimeFromPrev = (dt - dtPrev)
                minsFromPrev = divmod(elapsedTimeFromPrev.total_seconds(), 60)[0]

                latPrev = resp[i - 1][6]
                lonPrev = resp[i - 1][7]

                lat_lon_Curr = (lat, lon)
                lat_lon_Prev = (latPrev, lonPrev)

                dist = geopy.distance.distance(lat_lon_Prev, lat_lon_Curr).nm

                focPredsOpt.append((focPredsOptFoc[i] / 1440) * minsFromPrev)
            else:

                focPredsOpt.append(focPredsOptFoc[i] / 1440)

            totalDist += dist
        sumPredsOptFoc = np.sum(focPredsOpt)

        originalArrtlgDate = datetime.strptime(resp[len(resp) - 1][5].split(".")[0], "%Y-%m-%d %H:%M:%S")
        originalDeptlgDate = datetime.strptime(resp[0][5].split(".")[0], "%Y-%m-%d %H:%M:%S")
        elapsedTimeOptimization = (originalArrtlgDate - originalDeptlgDate)

        totalSailingTimeOpt = np.round(divmod(elapsedTimeOptimization.total_seconds(), 60)[0] / 60, 2)

        return totalSailingTimeOpt, stwOpt, sumPredsOptFoc, totalDist

    ####Main Controller Function for getting optimized Route ####
    def getOptimizedRouteFromNavigatorForLeg(self, leg, reta):

        legWOExt = leg.split(".")[0]
        ### build request.json
        # restrictETAs = [True, False]
        restrictETA = reta
        # for restrictETA in restrictETAs:
        requestBodyJSON = self.buildJsonForRoutingRequest(leg, restrictETA)
        requestBodyJSON = json.dumps(requestBodyJSON)

        responseID = self.requestWeatherRoutingOpt(requestBodyJSON)

        self.requestID = json.loads(responseID)['request_id']
        print("Request ID: " + self.requestID)

        responseWeatherRouting = self.getWeatherRoutingOpt()
        responseWeatherRouting = json.loads(responseWeatherRouting)
        print("Response: " + str(responseWeatherRouting))

        if responseWeatherRouting['foundSol'] == False:
            return -1
        else:
            ####  write first response to webAPI folder for breakRoute Algorithm to run
            with open(self.pathToWebAPIForPavlosBreakRoute + '/response_' + self.vessel + '_' + legWOExt + '.json',
                      'w') as json_file:
                json.dump(responseWeatherRouting, json_file)
            ###break route in segments of 1 hour ######
            # api call
            oo = self.get_ReproducedRoutePerH(legWOExt)
            oo = json.loads(oo)
            responseWeatherRouting['path'] = oo

            ####  write updated route locally
            if os.path.isdir('./routingResponses/') == False:
                os.mkdir('./routingResponses/')
            if os.path.isdir('./routingResponses/' + self.vessel ) == False:
                os.mkdir('./routingResponses/' + self.vessel )

            with open('./routingResponses/' + self.vessel + '/response_' + self.vessel + '_' + str(
                    reta) + '_' + legWOExt + '.json', 'w') as json_file:
                json.dump(responseWeatherRouting, json_file)

            print("Mapping response with WS . . .")
            self.mapResponseRoute(leg, 'optimized', reta)

            print("Evaluating routes from Neural DT . . .")
            self.evalRoutes(leg, "optimized", reta)

            print("Drawing routes . . . ")
            self.saveResponseRouteCoords(leg, reta)
            #self.drawRoutesOnMap(leg)
            return 1

    def evalRoutes(self, leg, type, reta):

        legWOExt = leg.split(".")[0]
        '''type of route : optimized OR initial'''
        if type == 'optimized':
            dataLeg = pd.read_csv('./routingResponses/' + self.vessel + '/mappedResponse_' + self.vessel + "_" + str(
                reta) + '_' + legWOExt + '.csv').values

            dfResponseAPI = self.ndt.callNeuralDTOnRoutes(self.vessel, dataLeg, )

            #dfNN = self.ndt.evaluateNeuralOnFile(dataLeg, self.vessel, 15, False, reta, leg)

    def mapResponseRoute(self, leg, type, reta):

        '''map route with Weather Service'''
        legWOExt = leg.split(".")[0]

        '''shutil.copy('/home/dimitris/Desktop/basic.csv',
                    './routingResponses/' + self.vessel + '/mappedResponse_' + self.vessel + '_' + str(
                        reta) + '_' + legWOExt + '.csv')

        return'''

        if type == 'optimized':

            f = open('./routingRequestsBody/' + self.vessel + '/request_' + self.vessel + '_' + str(
                reta) + '_' + legWOExt + '.json', )
            request = json.load(f)

            f = open('./routingResponses/' + self.vessel + '/response_' + self.vessel + '_' + str(
                reta) + '_' + legWOExt + '.json', )
            optRoute = json.load(f)
            # optRoute = json.loads(optRoute)
            route = optRoute['path']

            timestamps = []
            depDate = request['optimization_model']['voyage']['departDT']
            initBearing = None
            draft = request['optimization_model']['vessel']['draft']
            depDateFormated = datetime.strptime(depDate, '%Y-%m-%d %H:%M:%S')
            currDt = None
            bearings = []
            swhs = []
            relWindDirs = []
            windSpeeds = []
            currentSpeeds = []
            relCurrentDirs = []
            lats = []
            lons = []

            for i in range(0, len(route)):

                lat = route[i]['lat']
                lon = route[i]['lon']

                if i == 0:
                    latNext = route[i + 1]['lat']
                    lonNext = route[i + 1]['lon']
                    fwd_azimuth, back_azimuth, distance = geodesic.inv(lon, lat, lonNext, latNext, )
                if i > 0:
                    latPrev = route[i - 1]['lat']
                    lonPrev = route[i - 1]['lon']

                    lat_lon_Curr = (lat, lon)
                    lat_lon_Prev = (latPrev, lonPrev)

                    ##bearing calc
                    fwd_azimuth, back_azimuth, distance = geodesic.inv(lonPrev, latPrev, lon, lat)

                    fwd_azimuth = fwd_azimuth + 180 if fwd_azimuth < 0 else fwd_azimuth
                    # brng = Geodesic.WGS84.Inverse(latPrev, lonPrev, lat, lon)['azi1']

                    stw = route[i]['speed']
                    stwKmH = stw * 1.852
                    dist = geopy.distance.distance(lat_lon_Prev, lat_lon_Curr).km
                    timeElapsedH = dist / stwKmH
                    currDt = timestamps[i - 1] + timedelta(hours=timeElapsedH)

                currDt = depDateFormated if i == 0 else currDt

                windSpeed, relWindDir, swellSWH, relSwelldir, wavesSWH, relWavesdDir, combSWH, relCombWavesdDir, currentSpeed, relCurrentDir = \
                    dread.mapWeatherData(fwd_azimuth, currDt, np.round(lat), np.round(lon))

                if windSpeed == -1:
                    print("weather not found")

                windSpeeds.append(windSpeed)

                relWindDir = relWindDir - 180 if relWindDir > 180 else relWindDir
                # relCurrentDir = relCurrentDir - 180 if relCurrentDir > 180 else relCurrentDir

                relWindDirs.append(relWindDir)
                swhs.append(swellSWH)
                bearings.append(fwd_azimuth)
                timestamps.append(currDt)
                lats.append(lat)
                lons.append(lon)
                currentSpeeds.append(currentSpeed)
                relCurrentDirs.append(relCurrentDir)

            ##calculate stw from sovg response from pavlos and currents from weather service
            sovg = np.array([k['speed'] for k in route])

            stwFromCurrents = []
            for i in range(0,len(sovg)):
                currSpeedEffect = self.ndt.calcSTWFromCurrents(relCurrentDirs[i], currentSpeeds[i])
                stwFromCurrents.append(sovg[i] - currSpeedEffect)

            #stwFromCurrents = self.calculateSTWFromCurrents(bearings, relCurrentDirs, currentSpeeds, sovg)


            if os.path.isdir('./routingResponses/') == False:
                os.mkdir('./routingResponses/')
                os.mkdir('./routingResponses/' + self.vessel)

            with open('./routingResponses/'+self.vessel+'/mappedResponse_' + self.vessel + '_' +str(reta)+'_'+ legWOExt +'.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(
                    ['draft', 'windDir', 'ws', 'stwC', 'swh', 'dt', 'lat', 'lon', 'currSpeed', 'currDir',
                     'sovg', 'vslHeading'])
                for i in range(0, len(route)):
                    #if windSpeeds[i] == -1:
                        #continue
                    data_writer.writerow(
                        [draft, relWindDirs[i], windSpeeds[i], stwFromCurrents[i], swhs[i],timestamps[i], lats[i], lons[i],
                        currentSpeeds[i], relCurrentDirs[i], sovg[i], bearings[i] ])

    def calculateSTWFromCurrents(self, vslHeading, relCurrentDir, currentSpeedKnots, sovg):

        stwFromCurrents = []

        for i in range(0, len(sovg)):
            xs = np.cos(vslHeading[i]) * sovg[i]
            ys = np.sin(vslHeading[i]) * sovg[i]

            xc = np.cos(relCurrentDir[i]) * currentSpeedKnots[i]
            yc = np.sin(relCurrentDir[i]) * currentSpeedKnots[i]

            xNew = xs + xc
            yNew = ys + yc

            stwC = np.round(np.sqrt(xNew ** 2 + yNew ** 2), 2)
            stwFromCurrents.append(stwC)

        return np.array(stwFromCurrents).astype(float)

    ######################################################## ===> API CALLS
    @router.post("/")
    def requestWeatherRoutingOpt(self, requestBodyJSON):

        access_token = '123-456-789-123'
        response = requests.post(
            "http://10.2.4.54:8000/wrobwebapi/wrob/requestrouteoptimization",
            headers={"content-type": "application/json; charset=UTF-8",
                     'Authorization': 'Bearer {}'.format(access_token)},
            data=requestBodyJSON, )

        requestID = response.text

        return requestID

    @router.get("/")
    def getWeatherRoutingOpt(self):

        responseBody = ""
        while responseBody == "":
            response = requests.get(
                "http://10.2.4.54:8000/wrobwebapi/wrob/getrouteoptimizationdata/?requestID=" + self.requestID,

            )
            responseBody = response.text
        return responseBody

    @router.get("/")
    def get_ReproducedRoutePerH(self, leg):
        vslLeg = self.vessel + "_" + leg
        response = requests.get(
            "https://localhost:5004/BreakRoute/?vsLeg=" + vslLeg,
            verify=False)
        return response.text

    ############################################################## ===> API CALLS
    def plotOptimizationVSInitialWeather(self, leg, legNum, reta, notAvailable=None):

        plt.clf()
        if notAvailable == None:
            routingResp = pd.read_csv(
                './routingResponses/' + self.vessel + '/mappedResponse_' + self.vessel + '_' + str(
                    reta) + '_' + leg).values

            windSpeedResp = routingResp[:, 2]

            dtResp = list(
                map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').replace(second=0, microsecond=0) if str(
                    x).__contains__('.') else
                datetime.strptime(x, '%Y-%m-%d %H:%M:%S').replace(second=0, microsecond=0), routingResp[:, 5]))

            routingResp[:, 5] = dtResp

            initialRoute = pd.read_csv('./legs/' + self.vessel + '/' + leg).values

            initialRoute[:, 8] = initialRoute[:, 8] + ":00"
            # windSpeedInit = np.array([k for k in initialRoute if k[8] in dtResp])
            windSpeedInit = np.array(
                [min(initialRoute, key=lambda x: abs(datetime.strptime(x[8], '%Y-%m-%d %H:%M:%S') - k)) for k in dtResp ])


            dataWS = windSpeedInit[:,10]


            plt.plot(dataWS, label='Wind Speed Initial route Per Hour')
            plt.plot(windSpeedResp, label='Wind Speed Optimized route Per Hour', color='red')
            # plt.title("Leg: " +leg)
            plt.grid()
            plt.legend()
            plt.xlabel('Time (H)')
            plt.ylabel('Wind Speed (m/s)')

            # plt.show()
            plt.savefig(
                './Figures/' + self.company + '/' + self.vessel + "/" + "optimized_initial_" + str(reta) + '_' + str(
                    legNum) + '.eps', format='eps')

        elif notAvailable == True:

            # get source file name
            src = "./Figures/NA.eps"

            # get destination file name
            dest = './Figures/' + self.company + '/' + self.vessel + "/" + "optimized_initial_" + str(reta) + '_' + str(
                legNum) + '.eps'

            # rename source file name with destination file name

            # plt.show()
            shutil.copy(src, dest)
            # os.rename(src, dest)

    def extractInfoForWeatherComp(self, data, dataCorr, legInd=None):

        ## index sensorWS: 11
        ## index weatherServiceWS: 28
        wsIndC = 3 if legInd != None else 11
        stwIndC = 0 if legInd != None else 12

        wfS = data[:, 11].astype(float)  # / (1.944)
        print(np.max(wfS))
        wfS = np.round(wfS, 2)
        wfS = wfS[wfS < 30]
        wfDf = pd.DataFrame({'Wind Speed Sensor (m/s)': wfS})
        # sns.displot(wfDf, x="Wind Speed Sensor (m/s)")

        '''if legInd != None:
            plt.savefig('./Figures/'+self.company+'/' + self.vslFig + '/windSpeedSensorleg'+str(legInd)+'.eps', format='eps')
        else:
            plt.savefig('./Figures/' + self.company + '/' + self.vslFig + '/windSpeedSensor.eps',format='eps')'''

        wfSWS = data[:, 28].astype(float)
        wfSWS = np.round(wfSWS, 2)
        wfSWS = wfSWS[wfSWS > -1]
        wfSWSDf = pd.DataFrame({'Wind Speed NOA (m/s)': wfSWS})

        dict = {'source': [], 'Wind Speed (m/s)': []}
        for i in range(0, len(wfS)):
            dict['source'].append('sensor')
            dict['Wind Speed (m/s)'].append(wfS[i])

        for i in range(0, len(wfSWS)):
            dict['source'].append('NOA')
            dict['Wind Speed (m/s)'].append(wfSWS[i])

        dictDF = pd.DataFrame(dict)

        # plt.figure(figsize=(5, 5))

        g = sns.displot(data=dictDF, x="Wind Speed (m/s)", hue="source", fill=True, kind='kde',
                        palette=sns.color_palette('bright')[:2], height=5, aspect=1.5)
        g.fig.set_figwidth(5.27)
        g.fig.set_figheight(3.7)

        # plt.show()

        # sns.displot(wfSWSDf, x="Wind Speed NOA (m/s)")
        plt.xlim(min(wfS), max(wfS))
        '''if legInd != None:
            plt.savefig('./Figures/'+self.company+'/'+self.vslFig+'/windSpeedNOAleg'+str(legInd)+'.eps', format='eps')
        else:
            plt.savefig('./Figures/' + self.company + '/' + self.vslFig + '/windSpeedNOA.eps',
                        format='eps')'''

        if legInd != None:
            g.savefig('./Figures/' + self.company + '/' + self.vslFig + '/windSpeedWS_NOAleg' + str(legInd) + '.pdf',
                      format='pdf')
        else:
            g.savefig('./Figures/' + self.company + '/' + self.vslFig + '/windSpeedWS_NOA.pdf',
                      format='pdf')

        # plt.show()
        wsS_means = []
        wsN_means = []
        wsC_means = []
        wsS_std = []
        wsN_std = []
        wsC_std = []
        labels = []
        minSpeed = 10
        maxSpeed = 21
        i = minSpeed
        while i <= maxSpeed:

            wsSData = np.array([k for k in data if k[12] >= i - 0.5 and k[12] <= i + 0.5])
            wsNData = np.array([k for k in data if k[12] >= i - 0.5 and k[12] <= i + 0.5])
            wsCData = np.array([k for k in dataCorr if k[stwIndC] >= i - 0.5 and k[stwIndC] <= i + 0.5])

            if len(wsSData) > 0 and len(wsNData) > 0 and len(wsCData) > 0:
                wsS_means.append(np.mean(wsSData[:, 11]))
                wsN_means.append(np.mean(wsNData[:, 28]))
                wsC_means.append(np.mean(wsCData[:, wsIndC]))

                wsS_std.append(np.std(wsSData[:, 11]))
                wsN_std.append(np.std(wsNData[:, 28]))
                wsC_std.append(np.std(wsCData[:, wsIndC]))

                labels.append(str(i - 0.5) + " - " + str(i + 0.5))

            i += 1

        width = 0.4  # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots(figsize=(15, 8))
        X_axis = np.arange(len(labels))
        ax.bar(X_axis - 0.2, wsS_means, width, yerr=wsS_std, label='Wind Speed Sensor')

        ax.bar(X_axis + 0.2, wsN_means, width, yerr=wsN_std, label='Wind Speed NOA', color='green')

        ax.bar(X_axis + 0.4, wsC_means, width, yerr=wsC_std, label='Wind Speed Corrected', color='red')

        plt.xticks(X_axis, labels)

        ax.set_ylabel('Wind Speed (m/s)')
        ax.set_xlabel('STW (kt)')
        ax.set_title('Mean Wind Speed value / stw ranges for Sensor & NOA')
        ax.legend()
        if legInd != None:
            plt.savefig(
                './Figures/' + self.company + '/' + self.vslFig + '/windSpeedCompPerSpeedleg' + str(legInd) + '.eps',
                format='eps')
        else:
            plt.savefig('./Figures/' + self.company + '/' + self.vslFig + '/windSpeedCompPerSpeed.eps',
                        format='eps')
        # plt.show()

    def saveResponseRouteCoords(self, leg, reta):

        # Opening JSON file
        legWOExt = leg.split(".")[0]
        '''f = open('./routingResponses/' + self.vessel + '/response_' + self.vessel + '_' + str(
            reta) + '_' + legWOExt + '.json', )
        response = json.load(f)'''

        response = pd.read_csv('./routingResponses/' + self.vessel + '/mappedResponse_' + self.vessel + '_' + str(reta) + '_' + leg)
        # response = json.loads(response)
        path = response
        #path = response['path']
        lats = []
        lons = []
        for i in range(0, len(path)):
            lats.append(path['lat'][i])
            lons.append(path['lon'][i])

        if os.path.isdir('./coords/' + self.company + '/' + self.vessel ) == False:
            os.mkdir('./coords/' + self.company + '/' + self.vessel )

        with open('./coords/' + self.company + '/' + self.vessel + '/' + str(
                reta) + '_' + legWOExt + '_coords_response.csv', mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Latitude', 'Longitude'])
            for i in range(0, len(lats)):
                data_writer.writerow(
                    [lats[i], lons[i]])

        # coordsFile = pd.read_csv('./data/' + self.company + '/' + self.vessel + '/' + self.vessel + 'coords.csv').values

    def drawRoutesOnMap(self, leg, type, reta):

        legWOExt = leg.split(".")[0]

        coordsFileInit = pd.read_csv(
            './coords/' + self.company + '/' + self.vessel + '/' + str(reta) + '_' + legWOExt + '_coords_.csv').values

        coordsFile = pd.read_csv('./coords/' + self.company + '/' + self.vessel + '/' + str(
            reta) + '_' + legWOExt + '_coords_' + type + '.csv').values

        jsonCoords = {
            "type": "Feature",
            "properties": {"route": "",
                           },
            "geometry": {
                "type": "MultiLineString",
                "coordinates": [[

                ]],
                "properties": {
                    "color": "ff0000",
                    "slug": "adria-1"
                }
            }
        }

        # print(np.typename(coords[0][0]))
        for i in range(0, len(coordsFile)):
            jsonCoords['geometry']['coordinates'][0].append([float(coordsFile[i][1]), float(coordsFile[i][0])])

        destPoint = (coordsFileInit[len(coordsFileInit) - 1, 0], coordsFileInit[len(coordsFileInit) - 1, 1])
        origPoint = (coordsFileInit[0, 0], coordsFileInit[0, 1])

        self.plot_path(coordsFileInit[:, 0], coordsFileInit[:, 1], coordsFile[:, 0], coordsFile[:, 1], origPoint,
                       destPoint, leg)

        # with open('./data/'+self.company+'/'+self.vessel+'/'+self.vessel+'coords_'+type+'.json', 'w') as json_file:
        # json.dump(states, json_file)
        # Map = leafmap.Map(center=[0, 0], zoom=2, control_scale=True)
        # Map.add_geojson('./data/'+self.company+'/'+self.vessel+'/'+self.vessel+'coords_'+type+'.json', layer_name='leg')

        # embed_minimal_html('./map.html', views=[Map])
        # Map.to_html('./map.html')
        # Map.publish("sddeeded")
        # self.HTMLtoPNG(Map)

    def buildJsonForRoutingRequest(self, leg, reta):

        latInd = 8
        lonInd = 9
        bearInd = 6
        dtInd = 7

        legWOExt = leg.split(".")[0]

        # imo, type = self.extractImoType()

        tempRequestJson = self.requestJSON

        dataLeg = pd.read_csv('./correctedLegsForTR/' + self.vessel + '/' + leg).values

        tempRequestJson['vessel_code'] = self.imo
        tempRequestJson['optimization_model']['vessel']['vslCode'] = self.imo
        tempRequestJson['optimization_model']['vessel']['draft'] = np.round(np.mean(dataLeg[:, 1]), 2)
        tempRequestJson['optimization_model']['voyage']['departDT'] = dataLeg[0][dtInd]
        tempRequestJson['optimization_model']['voyage']['arriveDT'] = dataLeg[len(dataLeg) - 1][dtInd]

        tempRequestJson['optimization_model']['voyage']['restrictETA'] = reta
        tempRequestJson['optimization_model']['voyage']['minSpeedVoyage'] = 0
        tempRequestJson['optimization_model']['voyage']['maxSpeedVoyage'] = 0

        tempRequestJson['optimization_model']['options']['fuel'] = False
        tempRequestJson['optimization_model']['options']['ls'] = 0
        tempRequestJson['optimization_model']['options']['hs'] = 0
        tempRequestJson['optimization_model']['options']['time'] = False
        tempRequestJson['optimization_model']['options']['tce'] = 0
        tempRequestJson['optimization_model']['options']['weather'] = True

        legItemDep = {
            "lon": dataLeg[0][lonInd],
            "lat": dataLeg[0][latInd],
            "pCode": "",
            "pName": "",
            "speed": np.round(np.mean(dataLeg[:, 11]), 2),
            "gc": True,
            "locked": False,
            "waitHr": 0.0
        }

        legItemArr = {
            "lon": dataLeg[len(dataLeg) - 1][lonInd],
            "lat": dataLeg[len(dataLeg) - 1][latInd],
            "pCode": "",
            "pName": "",
            "speed": np.round(np.mean(dataLeg[:, 11]), 2),
            "gc": True,
            "locked": False,
            "waitHr": 0.0
        }

        tempRequestJson['optimization_model']['voyage']["path"] = []
        tempRequestJson['optimization_model']['voyage']["path"].append(legItemDep)
        tempRequestJson['optimization_model']['voyage']["path"].append(legItemArr)

        lats = []
        lons = []
        delIndices = []

        latLons = {'data': []}
        distFromPrev = 0
        distFromPrevRaw = 0
        i = 0
        listIs = []
        latDep = dataLeg[0][latInd]
        lonDep = dataLeg[0][lonInd]

        latArr = dataLeg[len(dataLeg) - 1][latInd]
        lonArr = dataLeg[len(dataLeg) - 1][lonInd]

        lat_lon_Dep = (latDep, lonDep)
        lat_lon_Arr = (latArr, lonArr)

        totalDist = geopy.distance.distance(lat_lon_Dep, lat_lon_Arr).km
        distTraveled = 0
        bearings = []
        i = 0
        ##route corrections
        for i in range(0, len(dataLeg)):

            lat = dataLeg[i][latInd]
            lon = dataLeg[i][lonInd]

            if i == 0:
                lats.append(lat)
                lons.append(lon)

                latNext = dataLeg[i + 1][latInd]
                lonNext = dataLeg[i + 1][lonInd]
                fwd_azimuth, back_azimuth, distance = geodesic.inv(lon, lat, lonNext, latNext)
                bearings.append(dataLeg[i][bearInd])

                continue

            latPrev = dataLeg[i - 1][latInd]
            lonPrev = dataLeg[i - 1][lonInd]

            fwd_azimuth, back_azimuth, distance = geodesic.inv(lonPrev, latPrev, lon, lat)

            bearings.append(fwd_azimuth)

            bearingPrev = bearings[i - 1]

            lat_lon_Prev = (latPrev, lonPrev)

            lat_lon_Curr = (lat, lon)

            stwPrev = dataLeg[i - 1][0]

            stwPrevKMperMin = (stwPrev) / 32.39741

            dt = datetime.strptime(dataLeg[i][dtInd], "%Y-%m-%d %H:%M:%S")
            dtPrev = datetime.strptime(dataLeg[i - 1][dtInd], "%Y-%m-%d %H:%M:%S")
            elapsedTimeFromPrev = (dt - dtPrev)
            minsFromPrev = np.round(divmod(elapsedTimeFromPrev.total_seconds(), 60)[0], 2)

            distanceKM = stwPrevKMperMin * minsFromPrev

            distFromPrevRaw = geopy.distance.distance(lat_lon_Prev, lat_lon_Curr).km

            destination = geopy.distance.distance(kilometers=distanceKM).destination((latPrev, lonPrev),
                                                                                     bearingPrev)
            latCorr, lonCorr = destination.latitude, destination.longitude

            if distFromPrevRaw > distanceKM:
                # or distFromTwoPrevRaw > 5 or distFromThreePrevRaw > 5 or (lat < latDep or lat > latArr) or (lon < lonDep or lon > lonArr)
                # delIndices.append(i)
                listIs.append(i)
                # dataLeg = np.delete(dataLeg, i, axis=0)

                # i = i + 1
                # continue
                lat = latCorr
                lon = lonCorr
                # continue
                dataLeg[i][latInd] = latCorr
                dataLeg[i][lonInd] = lonCorr

            # if distFromPrevRaw < 5 or i==0:
            item = {'lat': lat, 'lon': lon, "ind": i, 'distFromPrev': distFromPrev}
            latLons['data'].append(item)

            # if globe.is_ocean(lat,lon) == True:
            lats.append(lat)
            lons.append(lon)

            distTraveled += distanceKM
            i += 1
            '''legItem = {
                            "lon": dataLeg[i][7],
                            "lat": dataLeg[i][6],
                            "pCode": "",
                            "pName": "",
                            "speed": dataLeg[i][0],
                            "gc": True,
                            "locked": False,
                            "waitHr": 0.0
                        }
            tempRequestJson['optimization_model']['voyage']["path"].append(legItem)'''

            # if distFromPrevRaw < 5 or i == 0:
            # i = i + 1
        if lats[len(lats) - 1] != latDep:
            pass

        lats.append(latArr)
        lons.append(lonArr)
        # lats.append(dataLeg[len(dataLeg)-1][6])
        # lons.append(dataLeg[len(dataLeg)-1][7])
        print(len(listIs))
        # data = np.delete(data, indicesToBeDeleted, axis=0)
        if os.path.isdir('./coords/')==False:
            os.mkdir('./coords/')

        if os.path.isdir('./coords/' + self.company ) == False:
            os.mkdir('./coords/' + self.company )

        if os.path.isdir('./coords/' + self.company + '/' + self.vessel   ) == False:
            os.mkdir('./coords/' + self.company + '/' + self.vessel )

        with open('./coords/' + self.company + '/' + self.vessel + '/' + str(reta) + '_' + legWOExt + '_coords_.csv',
                  mode='w') as data:
            data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data_writer.writerow(['Latitude', 'Longitude'])
            for i in range(0, len(lats)):
                data_writer.writerow(
                    [lats[i], lons[i]])

        # self.drawCoordsOnMap("")

        # return
        if os.path.isdir('./routingRequestsBody/' ) == False:
            os.mkdir('./routingRequestsBody/')
        if os.path.isdir('./routingRequestsBody/' + self.vessel ) == False:
            os.mkdir('./routingRequestsBody/' + self.vessel )

        with open('./routingRequestsBody/' + self.vessel + '/request_' + self.vessel + '_' + str(
                reta) + '_' + legWOExt + '.json', 'w') as json_file:
            json.dump(tempRequestJson, json_file)

        return tempRequestJson

    def processLATLONToDraw(self, lat, long, latInit, longInit):

        indOfRef = 1000000000
        for i in range(0, len(long)):
            if i > 0:
                if abs(long[i] + long[i - 1]) < 1 and abs(long[i] + long[i - 1]) > 0:
                    indOfRef = i

            if i >= indOfRef:
                long[i] += 360

        indOfRef = 1000000000
        for i in range(0, len(longInit)):
            if i > 0:
                if abs(longInit[i] + longInit[i - 1]) < 1 and abs(longInit[i] + longInit[i - 1]) > 0:
                    indOfRef = i

            if i >= indOfRef:
                longInit[i] += 360

        '''for i in range(0, len(lat)):
            if i > 0:
                if abs(lat[i] + lat[i - 1]) < 1 and abs(lat[i] + lat[i - 1]) > 0:
                    indOfRef = i

            if i >= indOfRef:
                lat[i] += 180

        indOfRef = 1000000000
        for i in range(0, len(latInit)):
            if i > 0:
                if abs(latInit[i] + latInit[i - 1]) < 1 and abs(latInit[i] + latInit[i - 1]) > 0:
                    indOfRef = i

            if i >= indOfRef:
                latInit[i] += 180'''

        return long, longInit

    def plot_path(self, latInit, longInit, lat, long, origin_point, destination_point, leg):

        zoomMap = None
        if len(longInit) > 10000:
            zoomMap = 2

        if len(longInit) < 10000:
            zoomMap = 4

        if len(longInit) < 1000:
            zoomMap = 6

        #long, longInit = self.processLATLONToDraw(lat, long, latInit, longInit)

        legWOExt = leg.split(".")[0]
        """
        Given a list of latitudes and longitudes, origin
        and destination point, plots a path on a map

        Parameters
        ----------
        lat, long: list of latitudes and longitudes
        origin_point, destination_point: co-ordinates of origin
        and destination
        Returns
        -------
        Nothing. Only shows the map.
        """
        layout = go.Layout(
            autosize=False,
            width=1400,
            height=1000,

            xaxis=go.layout.XAxis(linecolor='black',
                                  linewidth=1,
                                  mirror=True),

            yaxis=go.layout.YAxis(linecolor='black',
                                  linewidth=1,
                                  mirror=True),

            margin=go.layout.Margin(
                l=50,
                r=50,
                b=50,
                t=50,
                pad=4
            ),

        )

        # adding the lines joining the nodes optimized route
        fig = go.Figure(go.Scattermapbox(
            name="Optimization",
            mode="lines",
            lon=long,
            lat=lat,
            marker={'size': 10},
            line=dict(width=3.5, color='red'), ), layout=layout)

        # adding the lines joining the nodes initial route route
        fig.add_trace(go.Scattermapbox(
            name="Original",
            mode="lines",
            lon=longInit,
            lat=latInit,
            marker={'size': 10},
            line=dict(width=3.5, color='blue')))

        # adding source marker
        fig.add_trace(go.Scattermapbox(
            name="Departure",
            mode="markers",
            lon=[origin_point[1]],
            lat=[origin_point[0]],
            marker={'size': 15, 'color': "green"}))

        # adding destination marker
        fig.add_trace(go.Scattermapbox(
            name="Arrival",
            mode="markers",
            lon=[destination_point[1]],
            lat=[destination_point[0]],
            marker={'size': 15, 'color': 'black'}))

        # getting center for plots:
        lat_center = np.mean(lat)
        long_center = np.mean(long)
        # defining the layout using mapbox_style
        fig.update_layout(mapbox_style="stamen-terrain",
                          mapbox_center_lat=30, mapbox_center_lon=-80)
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                          mapbox={
                              'center': {'lat': lat_center,
                                         'lon': long_center},
                              'zoom': zoomMap,
                              'style': "stamen-terrain",
                          })
        # fig.show()
        if str(leg).__contains__(" "):
            leg1 = legWOExt.split(" ")[0]
            leg2 = legWOExt.split(" ")[1]
        legF = leg1 + "_" + leg2 if str(leg).__contains__(" ") else legWOExt

        try:
            fig.write_image('./Figures/' + self.company + '/' + self.vslFig + '/map_' + legF + '.pdf')
        except Exception as e:
            print(e)
            print("Error writing map " + self.vslFig + '/map_' + legF + '.pdf')

            shutil.copy('./Figures/NA.pdf', './Figures/' + self.company + '/' + self.vslFig + '/map_' + legF + '.pdf')
        # fig.show()


def main():
    server = '10.2.5.80'
    sid = 'OR12'
    password = 'shipping'
    usr = 'shipping'
    company = 'DANAOS'
    vessel = r'''NERVAL'''
    genRep = GenerateReport(company, vessel, server, sid, password, usr)

    # plotOptimizationVSInitialWeather("MIAMI_SUEZ_leg.csv", 1)
    # leg = "PUSAN_PANAMA_leg.csv"
    # genRep.buildJsonForRoutingRequest(leg)
    # data = pd.read_csv('./data/' + company + '/' + vessel + '/mappedData.csv').values
    # genRep.extractInfoForWeatherComp(data,data)
    # genRep.drawRoutesOnMap(leg, "response")
    # genRep.plotOptimizationVSInitialWeather("", 1, False, True)
    # genRep.drawRoutesOnMap('KINGSTON_NINGBO_leg', 'response', False)
    # return
    #genRep.extractImoType()
    #genRep.buildJsonForRoutingRequest('SHEKOU_NINGBO_leg.csv', True)
    genRep.generate()


if __name__ == "__main__":
    main()
