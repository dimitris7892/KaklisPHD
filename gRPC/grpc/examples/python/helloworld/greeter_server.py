# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging
import numpy as np
import csv
import grpc
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import helloworld_pb2
import helloworld_pb2_grpc
import sys , time , datetime
import time

class Greeter(helloworld_pb2_grpc.GreeterServicer):

  def __init__(self):

    self.CountServer = 0
    self.deployedModels = []
    for i in range(0, 4):
      currModeler = keras.models.load_model('../../../../../Danaos_ML_Project/DeployedModels/estimatorCl_' + str(i) + '.h5')
      self.deployedModels.append(currModeler)
    self.currModelerGen = keras.models.load_model('../../../../../Danaos_ML_Project/DeployedModels/estimatorCl_Gen.h5')

    self.partitionsX = []
    self.partitionsY = []
    for cl in range(0, 4):
      data = pd.read_csv('../../../../../Danaos_ML_Project/DeployedModels/cluster_' + str(cl) + '_.csv')
      self.partitionsX.append(readClusteredLarosDataFromCsvNew(data))

    for cl in range(0, 4):
      data = pd.read_csv('../../../../../Danaos_ML_Project/DeployedModels/cluster_foc' + str(cl) + '_.csv')
      self.partitionsY.append(readClusteredLarosDataFromCsvNew(data))

    for cl in range(0, 4):
      data = pd.read_csv('../../../../../Danaos_ML_Project/DeployedModels/cluster_foc' + str(cl) + '_.csv')
      self.partitionsY.append(readClusteredLarosDataFromCsvNew(data))

    self.dataCsvModels=[]
    for modelId in range(0, 4):
      self.dataCsvModel = []
      csvModels = ['../../../../../Danaos_ML_Project/trainedModels/model_' + str(modelId) + '_.csv']
      for csvM in csvModels:
        if csvM != '../../../../../Danaos_ML_Project/trainedModels/model_' + str(modelId) + '_.csv':
          continue
      # id = csvM.split("_")[ 1 ]
      # piecewiseFunc = [ ]

        with open(csvM) as csv_file:
          data = csv.reader(csv_file, delimiter=',')
          for row in data:
            self.dataCsvModel.append(row)
        self.dataCsvModels.append(self.dataCsvModel)


    csvM = '../../../../../Danaos_ML_Project/trainedModels/model_Gen_.csv'
    self.dataCsvModel = []
    with open(csvM) as csv_file:
      data = csv.reader(csv_file, delimiter=',')
      for row in data:
        self.dataCsvModel.append(row)
    self.dataCsvModels.append(self.dataCsvModel)

    self.meanOfClusters = []
    for cl in range(0, 4):
      self.meanOfClusters.append(np.mean(self.partitionsX[cl],axis=0))

  def getFitnessOfPoint(self,partitions, cluster, point):
    # return distance.euclidean(np.mean(partitions[cluster], axis=0) , point)
    return 1 / (1 + np.linalg.norm(self.meanOfClusters[cluster] - point))
    # return 1 / (1 + np.linalg.norm(np.mean(np.array(partitions[cluster][:, 4].reshape(-1, 1))) - np.array(point[0][4])))
    # return 1 / (1 + np.linalg.norm(np.mean(np.array(partitions[ cluster ][ :,1 ].reshape(-1, 1))) - np.array(point[ 0 ][ 1 ] )))

    # return 1 / (1 + np.linalg.norm(np.mean(partitions[cluster], axis=0) - point))
    # return 1 / (1 + np.linalg.norm(np.mean(partitions[cluster][:,1],axis=0) -point[:,1]))

  def getBestPartitionForPoint(self,point, partitions):
    # mBest = None
    mBest = None
    dBestFit = 0
    #fits = {"data": []}
    # For each model
    for m in range(0, len(partitions)):
      # If it is a better for the point
      dCurFit = self.getFitnessOfPoint(partitions, m, point)
      '''fit = {}
      fit["fit"] = dCurFit
      fit["id"] = m
      fits["data"].append(fit)'''

      if dCurFit > dBestFit:
        # Update the selected best model and corresponding fit
        dBestFit = dCurFit
        mBest = m

    if mBest == None:
      return 0, 0
    else:
      return mBest, dBestFit

  def extractFunctionsFromSplines(self, x0, x1, x2, x3, x4, x5, x6, modelId):
    piecewiseFunc = []
    #csvModels = ['../../../../../Danaos_ML_Project/trainedModels/model_' + str(modelId) + '_.csv']
    #for csvM in csvModels:
      #if csvM != '../../../../../Danaos_ML_Project/trainedModels/model_' + str(modelId) + '_.csv':
        #continue
      # id = csvM.split("_")[ 1 ]
      # piecewiseFunc = [ ]

    #with open(csvM) as csv_file:
    #data = csv.reader(csv_file, delimiter=',')
    data = self.dataCsvModels[modelId] if modelId !='Gen' else self.dataCsvModels[len(self.dataCsvModels)-1]
    #id = modelId
    for row in data:
          # for d in row:
          if [w for w in row if w == "Basis"].__len__() > 0:
            continue
          if [w for w in row if w == "(Intercept)"].__len__() > 0:
            self.interceptsGen = float(row[1])
            continue

          if row.__len__() == 0:
            continue
          d = row[0]
          # if self.count == 1:
          # self.intercepts.append(float(row[1]))

          if d.split("*").__len__() == 1:
            split = ""
            try:
              split = d.split('-')[0][2:3]
              if split != "x":
                split = d.split('-')[1]
                num = float(d.split('-')[0].split('h(')[1])
                # if id == id:
                # if float(row[ 1 ]) < 10000:
                try:
                  # piecewiseFunc.append(
                  # tf.math.multiply(tf.cast(tf.math.less(x, num), tf.float32),
                  # (num - inputs)))
                  if split.__contains__("x0"):
                    piecewiseFunc.append((num - x0))  # * float(row[ 1 ]))
                  if split.__contains__("x1"):
                    piecewiseFunc.append((num - x1))  # * float(row[ 1 ]))
                  if split.__contains__("x2"):
                    piecewiseFunc.append((num - x2))  # * float(row[ 1 ]))
                  if split.__contains__("x3"):
                    piecewiseFunc.append((num - x3))  # * float(row[ 1 ]))
                  if split.__contains__("x4"):
                    piecewiseFunc.append((num - x4))  # * float(row[ 1 ]))
                  if split.__contains__("x5"):
                    piecewiseFunc.append((num - x5))  # * float(row[ 1 ]))
                  if split.__contains__("x6"):
                    piecewiseFunc.append((num - x6))  # * float(row[ 1 ]))
                # if id ==  self.modelId:
                # inputs = tf.where(x >= num, float(row[ 1 ]) * (inputs - num), inputs)
                except:
                  dc = 0
              else:
                ##x0 or x1
                split = d.split('-')[0]
                num = float(d.split('-')[1].split(')')[0])
                # if id == id:
                # if float(row[ 1 ]) < 10000:
                try:
                  if split.__contains__("x0"):
                    piecewiseFunc.append((x0 - num))  # * float(row[ 1 ]))
                  if split.__contains__("x1"):
                    piecewiseFunc.append((x1 - num))  # * float(row[ 1 ]))
                  if split.__contains__("x2"):
                    piecewiseFunc.append((x2 - num))  # * float(row[ 1 ]))
                  if split.__contains__("x3"):
                    piecewiseFunc.append((x3 - num))  # * float(row[ 1 ]))
                  if split.__contains__("x4"):
                    piecewiseFunc.append((x4 - num))  # * float(row[ 1 ]))
                  if split.__contains__("x5"):
                    piecewiseFunc.append((x5 - num))  # * float(row[ 1 ]))
                  if split.__contains__("x6"):
                    piecewiseFunc.append((x6 - num))  # * float(row[ 1 ]))

                  # piecewiseFunc.append(
                  # tf.math.multiply(tf.cast(tf.math.greater(x, num), tf.float32),
                  # (inputs - num)))
                # if id == self.modelId:
                # inputs = tf.where(x <= num, float(row[ 1 ]) * (num - inputs), inputs)
                except:
                  dc = 0
            except:
              # if id == id:
              # if float(row[ 1 ]) < 10000:
              try:
                piecewiseFunc.append(x0)

                # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                # (inputs)))

                # inputs = tf.where(x >= 0, float(row[ 1 ]) * inputs, inputs)
              # continue
              except:
                dc = 0

          else:
            funcs = d.split("*")
            nums = []
            flgFirstx = False
            flgs = []
            for r in funcs:
              try:
                if r.split('-')[0][2] != "x":
                  flgFirstx = True
                  nums.append(float(r.split('-')[0].split('h(')[1]))

                else:
                  nums.append(float(r.split('-')[1].split(')')[0]))

                flgs.append(flgFirstx)
              except:
                flgFirstx = False
                flgs = []
                split = d.split('-')[0][2]
                try:
                  if d.split('-')[0][2] == "x":
                    # if id == id:
                    # if float(row[ 1 ]) < 10000:
                    try:

                      if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                        "x1"):
                        split = "x1"
                      if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                        "x0"):
                        split = "x0"
                      if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                        "x1"):
                        split = "x01"
                      if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                        "x0"):
                        split = "x10"

                      if split == "x0":
                        piecewiseFunc.append(x0 * (x0 - nums[0]) * float(row[1]))
                      elif split == "x1":
                        piecewiseFunc.append(x1 * (x1 - nums[0]) * float(row[1]))
                      elif split == "x01":
                        piecewiseFunc.append(x0 * (x1 - nums[0]) * float(row[1]))
                      elif split == "x10":
                        piecewiseFunc.append(x1 * (x0 - nums[0]) * float(row[1]))
                      # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                      # (inputs) * (
                      # inputs - nums[ 0 ])))

                      # inputs = tf.where(x >= 0,
                      # float(row[ 1 ]) * (inputs) * (inputs - nums[ 0 ]), inputs)
                    except:
                      dc = 0

                  else:
                    flgFirstx = True
                    # if id == id:
                    # if float(row[ 1 ]) < 10000:
                    try:

                      if d.split("-")[0].__contains__("x1") and d.split("-")[
                        1].__contains__("x1"):
                        split = "x1"
                      if d.split("-")[0].__contains__("x0") and d.split("-")[
                        1].__contains__("x0"):
                        split = "x0"
                      if d.split("-")[0].__contains__("x0") and d.split("-")[
                        1].__contains__("x1"):
                        split = "x01"
                      if d.split("-")[0].__contains__("x1") and d.split("-")[
                        1].__contains__("x0"):
                        split = "x10"

                      if split == "x0":
                        piecewiseFunc.append(x0 * (nums[0] - x0) * float(row[1]))
                      elif split == "x1":
                        piecewiseFunc.append(x1 * (nums[0] - x1) * float(row[1]))
                      elif split == "x01":
                        piecewiseFunc.append(x0 * (nums[0] - x1) * float(row[1]))
                      elif split == "x10":
                        piecewiseFunc.append(x1 * (nums[0] - x0) * float(row[1]))

                      # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                      # (inputs) * (
                      # nums[ 0 ] - inputs)))

                      # inputs = tf.where(x > 0 ,
                      # float(row[ 1 ]) * (inputs) * (nums[ 0 ] - inputs),inputs)
                      flgs.append(flgFirstx)
                    except:
                      dc = 0

                except:
                  # if id == id:
                  # if float(row[ 1 ]) < 10000:
                  try:
                    piecewiseFunc.append(x0)

                    # piecewiseFunc.append(tf.math.multiply(tf.cast(x, tf.float32),
                    # (inputs)))

                    # inputs = tf.where(x >= 0, float(row[ 1 ]) * (inputs), inputs)
                  except:
                    dc = 0
            try:
              # if id == id:
              if flgs.count(True) == 2:
                # if float(row[ 1 ])<10000:
                try:

                  if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                    "x1"):
                    split = "x1"
                  if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                    "x0"):
                    split = "x0"
                  if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                    "x1"):
                    split = "x01"
                  if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                    "x0"):
                    split = "x10"

                  if split == "x0":
                    piecewiseFunc.append((nums[0] - x0) * (nums[1] - x0) * float(row[1]))
                  elif split == "x1":
                    piecewiseFunc.append((nums[0] - x1) * (nums[1] - x1) * float(row[1]))
                  elif split == "x01":
                    piecewiseFunc.append((nums[0] - x0) * (nums[1] - x1) * float(row[1]))
                  elif split == "x10":
                    piecewiseFunc.append((nums[0] - x1) * (nums[1] - x0) * float(row[1]))

                  # piecewiseFunc.append(tf.math.multiply(tf.cast(
                  # tf.math.logical_and(tf.math.less(x, nums[ 0 ]),
                  # tf.math.less(x, nums[ 1 ])), tf.float32),
                  # (nums[ 0 ] - inputs) * (
                  # nums[ 1 ] - inputs)))

                  # inputs = tf.where(x < nums[0] and x < nums[1],
                  # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                  # nums[ 1 ] - inputs), inputs)
                except:
                  dc = 0

              elif flgs.count(False) == 2:
                # if float(row[ 1 ]) < 10000:
                try:
                  if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                    "x1"):
                    split = "x1"
                  if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                    "x0"):
                    split = "x0"
                  if d.split("-")[0].__contains__("x0") and d.split("-")[1].__contains__(
                    "x1"):
                    split = "x01"
                  if d.split("-")[0].__contains__("x1") and d.split("-")[1].__contains__(
                    "x0"):
                    split = "x10"

                  if split == "x0":
                    piecewiseFunc.append((x0 - nums[0]) * (x0 - nums[1]) * float(row[1]))
                  elif split == "x1":
                    piecewiseFunc.append((x1 - nums[0]) * (x1 - nums[1]) * float(row[1]))
                  elif split == "x01":
                    piecewiseFunc.append((x0 - nums[0]) * (x1 - nums[1]) * float(row[1]))
                  elif split == "x10":
                    piecewiseFunc.append((x1 - nums[0]) * (x0 - nums[1]) * float(row[1]))
                  # inputs = tf.where(x > nums[ 0 ] and x > nums[ 1 ],
                  # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                  # inputs - nums[ 1 ]), inputs)
                except:
                  dc = 0
              else:
                try:
                  if flgs[0] == False:
                    if nums.__len__() > 1:
                      # if float(row[ 1 ]) < 10000:
                      try:
                        if d.split("-")[0].__contains__("x1") and d.split("-")[
                          2].__contains__("x1"):
                          split = "x1"
                        if d.split("-")[0].__contains__("x0") and d.split("-")[
                          2].__contains__("x0"):
                          split = "x0"
                        if d.split("-")[0].__contains__("x0") and d.split("-")[
                          2].__contains__("x1"):
                          split = "x01"
                        if d.split("-")[0].__contains__("x1") and d.split("-")[
                          2].__contains__("x0"):
                          split = "x10"

                        if split == "x0":
                          piecewiseFunc.append(
                            (x0 - nums[0]) * (nums[1] - x0) * float(row[1]))
                        elif split == "x1":
                          piecewiseFunc.append(
                            (x1 - nums[0]) * (nums[1] - x1) * float(row[1]))
                        elif split == "x01":
                          piecewiseFunc.append(
                            (x0 - nums[0]) * (nums[1] - x1) * float(row[1]))
                        elif split == "x10":
                          piecewiseFunc.append(
                            (x1 - nums[0]) * (nums[1] - x0) * float(row[1]))




                      # inputs = tf.where(x > nums[ 0 ] and x < nums[ 1 ],
                      # float(row[ 1 ]) * (inputs - nums[ 0 ]) * (
                      # nums[ 1 ] - inputs), inputs)
                      except:
                        dc = 0
                    else:
                      # if float(row[ 1 ]) < 10000:
                      try:
                        if d.split("-")[0].__contains__("x1") and d.split("-")[
                          1].__contains__("x1"):
                          split = "x1"
                        if d.split("-")[0].__contains__("x0") and d.split("-")[
                          1].__contains__("x0"):
                          split = "x0"
                        if d.split("-")[0].__contains__("x0") and d.split("-")[
                          1].__contains__("x1"):
                          split = "x01"
                        if d.split("-")[0].__contains__("x1") and d.split("-")[
                          1].__contains__("x0"):
                          split = "x10"

                        piecewiseFunc.append((x0 - nums[0]) * float(row[1]))

                        # inputs = tf.where(x > nums[0],
                        # float(row[ 1 ]) * (inputs - nums[ 0 ]), inputs)
                      except:
                        dc = 0
                  else:
                    if nums.__len__() > 1:
                      # if float(row[ 1 ]) < 10000:
                      try:
                        if d.split("-")[0].__contains__("x1") and d.split("-")[
                          1].__contains__("x1"):
                          split = "x1"
                        if d.split("-")[0].__contains__("x0") and d.split("-")[
                          1].__contains__("x0"):
                          split = "x0"
                        if d.split("-")[0].__contains__("x0") and d.split("-")[
                          1].__contains__("x1"):
                          split = "x01"
                        if d.split("-")[0].__contains__("x1") and d.split("-")[
                          1].__contains__("x0"):
                          split = "x10"

                        if split == "x0":
                          piecewiseFunc.append(
                            (nums[0] - x0) * (x0 - nums[1]) * float(row[1]))
                        elif split == "x1":
                          piecewiseFunc.append(
                            (nums[0] - x1) * (x1 - nums[1]) * float(row[1]))
                        elif split == "x01":
                          piecewiseFunc.append(
                            (nums[0] - x0) * (x1 - nums[1]) * float(row[1]))
                        elif split == "x10":
                          piecewiseFunc.append(
                            (nums[0] - x1) * (x0 - nums[1]) * float(row[1]))
                        # inputs = tf.where(x < nums[ 0 ] and x > nums[1],
                        # float(row[ 1 ]) * (nums[ 0 ] - inputs) * (
                        # inputs - nums[ 1 ]), inputs)
                      except:
                        dc = 0
                    else:
                      # if float(row[ 1 ]) < 10000:
                      try:

                        if d.split("-")[0].__contains__("x1") and d.split("-")[
                          1].__contains__("x1"):
                          split = "x1"
                        if d.split("-")[0].__contains__("x0") and d.split("-")[
                          1].__contains__("x0"):
                          split = "x0"
                        if d.split("-")[0].__contains__("x0") and d.split("-")[
                          1].__contains__("x1"):
                          split = "x01"
                        if d.split("-")[0].__contains__("x1") and d.split("-")[
                          1].__contains__("x0"):
                          split = "x10"

                        if split == "x0":
                          piecewiseFunc.append((x0 - nums[0]) * float(row[1]))
                        elif split == "x1":
                          piecewiseFunc.append((x1 - nums[0]) * float(row[1]))
                        # piecewiseFunc.append(tf.math.multiply(tf.cast(
                        # tf.math.less(x, nums[ 0 ]), tf.float32),
                        # (
                        # inputs - nums[ 0 ])))

                        # inputs = tf.where(x < nums[ 0 ],
                        # float(row[ 1 ]) * (
                        # inputs - nums[ 0 ]), inputs)
                      except:
                        dc = 0
                except:
                  dc = 0
            except:
              dc = 0

    return piecewiseFunc

  def GivePrediction(self, request, context):

        start_time = time.time()

        valClient  = request.name
        print(datetime.datetime.now().time())
        val = str(valClient).split("[")[1].split("]")
        values = val[0].split(",")
        vectorValues = np.array( [float(values[0]),float(values[1]),float(values[2]),float(values[3]),float(values[4]),float(values[5]),float(values[6])])
        pPoint = vectorValues

        ind, fit = self.getBestPartitionForPoint(pPoint, self.partitionsX)

        print("--- %s seconds --- CLASSIFY POINT IN CLUSTER" % (time.time() - start_time))

        start_time = time.time()

        vector = self.extractFunctionsFromSplines(pPoint[0], pPoint[1], pPoint[2], pPoint[3], pPoint[4],
                                                  pPoint[5], pPoint[6], ind)
        XSplineVector = np.append(pPoint, vector)
        XSplineVector = XSplineVector.reshape(-1, XSplineVector.shape[0])
        print("--- %s seconds --- BUILD ENRICHED VECTOR" % (time.time() - start_time))

        if self.CountServer ==0 :
          vector = self.extractFunctionsFromSplines(pPoint[0], pPoint[1], pPoint[2], pPoint[3], pPoint[4],
                                                    pPoint[5], pPoint[6], 'Gen')
          XSplineGenVector = np.append(pPoint, vector)
          self.XSplineGenVector = XSplineGenVector.reshape(-1, XSplineGenVector.shape[0])

        start_time = time.time()
        currModeler = self.deployedModels[ind]
        currModelerGen = self.currModelerGen
        import tensorflow as tf

        prediction = (abs(currModeler.predict(XSplineVector)) + currModelerGen.predict(self.XSplineGenVector)) / 2

        print("--- %s seconds --- PREDICTION" % (time.time() - start_time))

        self.CountServer =+1
        return helloworld_pb2.HelloReply(message=str(round(prediction[0][0],3)))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()

    server.wait_for_termination()

def readClusteredLarosDataFromCsvNew(data):
      # Load file
      dtNew = data.values[0:, :].astype(float)
      dtNew = dtNew[~np.isnan(dtNew).any(axis=1)]
      seriesX = dtNew
      UnseenSeriesX = dtNew
      return UnseenSeriesX

if __name__ == '__main__':
  ##xxx
  logging.basicConfig()
  serve()
