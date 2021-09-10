import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.datasets import load_breast_cancer

sns.set()


def get_seaborn_plot_as_bytes(data, feature_name):
    """
    Returns a seaborn plot as bytes
    :param data:
    :param feature_names: On which features the seaborn plot is on
    :return: io.BytesIO
    """
    #corr = data[list(feature_names)].corr(method='pearson')


    sns.displot(data, x=feature_name)

    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image


def get_foc_plot_as_bytes(data,):
    """
    Returns a matplotlib plot as bytes
    :param data:
    :param feature_names: On which features the matplotlib plot is on
    :return: io.BytesIO
    """
    trData = data.values[5:]
    fig, ax1 = plt.subplots(figsize=(15,10))
    trDataPLot = trData[trData[:, 7].argsort()].astype(float)
    raw = np.array([k for k in trDataPLot if
                    float(k[2]) >= 0 and float(k[4]) >= 0 and float(k[7]) > 7 and float(k[3]) <= 11 and float(
                        k[5]) > 0])

    i = min(trData[:,7])
    maxSpeed = max(trData[:,7])
    sizesSpeed = []
    speed = []
    avgActualFoc = []
    while i <= maxSpeed:
        # workbook._sheets[sheet].insert_rows(k+27)

        speedArray = np.array([k for k in raw if float(k[7]) >= i - 0.25 and float(k[7]) <= i + 0.25])

        if speedArray.__len__() > 1:
            sizesSpeed.append(speedArray.__len__())
            speed.append(i)
            avgActualFoc.append(np.mean(speedArray[:, 5]))
        i += 0.5


    minSpeed = min(trData[:,8])
    maxSpeed = max(trData[:,8])
    raw = np.array([k for k in trDataPLot if float(k[16]) >= 0 and float(k[24]) >= 0 and float(k[6]) > minSpeed and
                    float(k[6]) <= maxSpeed and float(k[9]) > 0])

    i = minSpeed
    maxSpeed = maxSpeed
    sizesSpeed = []
    speed = []
    avgActualFoc = []

    minActualFoc = []
    maxActualFoc = []
    stdActualFoc = []

    avgActualFocPerMile = []
    minActualFocPerMile = []
    maxActualFocPerMile = []
    stdActualFocPerMile = []
    while i <= maxSpeed:
        # workbook._sheets[sheet].insert_rows(k+27)

        speedArray = np.array([k for k in raw if float(k[6]) >= i - 0.25 and float(k[6]) <= i + 0.25])

        if speedArray.__len__() > 1:
            sizesSpeed.append(speedArray.__len__())
            speed.append(i)
            avgActualFoc.append(np.mean(speedArray[:, 9]))
            minActualFoc.append(np.min(speedArray[:, 9]))
            maxActualFoc.append(np.max(speedArray[:, 9]))
            stdActualFoc.append(np.std(speedArray[:, 9]))

            avgActualFocPerMile.append(np.mean(speedArray[:, 7]))
            minActualFocPerMile.append(np.min(speedArray[:, 7]))
            maxActualFocPerMile.append(np.max(speedArray[:, 7]))
            stdActualFocPerMile.append(np.std(speedArray[:, 7]))
        i += 0.5

    minDeviation = abs(np.array(avgActualFoc) - np.array(minActualFoc))
    maxDeviation = abs(np.array(avgActualFoc) - np.array(maxActualFoc))

    plt.fill_between(speed, np.array(avgActualFoc) - minDeviation, np.array(avgActualFoc) +
                     maxDeviation, color='gray', alpha=0.2,
                     label='Raw FOC variance  ' + str(np.round(np.var(raw[:, 9]), 2)) + " STD: " + str(
                         np.round(np.std(raw[:, 9]), 2)))

    # plt.plot(raw[:,3],raw[:,5],label='Raw FOC',c='orange',)

    plt.plot(speed, np.array(avgActualFoc) + 2 * np.array(stdActualFoc), '--', label='+2S', lw=3, zorder=20,
             color='green')
    plt.plot(speed, np.array(avgActualFoc) - 2 * np.array(stdActualFoc), '--', label='-2S', lw=3, zorder=20,
             color='green')

    trDataPLot = trData[trData[:, 3].argsort()].astype(float)
    for i in range(0, len(trDataPLot)):
        trDataPLot[i] = np.mean(trDataPLot[i:i + 15], axis=0)

    movingAvg = np.array([k for k in trDataPLot if float(k[16]) >= 0 and float(k[24]) >= 0 and float(k[6]) > minSpeed and
                    float(k[6]) <= maxSpeed and float(k[9]) > 0])
    #fig, axes = plt.subplots
    plt.plot(movingAvg[:, 6], movingAvg[:, 9], label='Moving AVG (15 min)', color='blue', )

    plt.scatter(speed, avgActualFoc, s=np.array(sizesSpeed) / 10, c='red', zorder=20)

    plt.plot(speed, avgActualFoc, label='Mean Foc / speed', lw=3, zorder=20, color='red')
    plt.legend()
    plt.grid()
    plt.xlabel('speed (knots)')
    plt.ylabel('FOC (MT/day)')


    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    #bytes_image.seek(0)
    #return bytes_image
    figure = FigureCanvas(fig).print_png(bytes_image)
    return bytes_image