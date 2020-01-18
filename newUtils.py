with open('./windSpeedMapped.csv', mode='w') as dataw:
    for k in range(0, len(windSpeedVmapped)):
        data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([windSpeedVmapped[k] if windSpeedVmapped[k] == None else windSpeedVmapped[k][0]])
with open('./windDirMapped.csv', mode='w') as dataw:
    for k in range(0, len(windDirsVmapped)):
        data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([windDirsVmapped[k] if windDirsVmapped[k] == None else windDirsVmapped[k][0]])

with open('./draftsMapped.csv', mode='w') as dataw:
    for k in range(0, len(draftsVmapped)):
        data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([draftsVmapped[k][0]])

with open('./blFlagsMapped.csv', mode='w') as dataw:
    for k in range(0, len(ballastLadenFlagsVmapped)):
        data_writer = csv.writer(dataw, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow([ballastLadenFlagsVmapped[k][0]])

windSpeedVmapped = []
windDirsVmapped = []
draftsVmapped = []
ballastLadenFlagsVmapped = []

with open('./windSpeedMapped.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        try:
            windSpeedVmapped.append(float(row[0]))
        except:
            x = 0
with open('./windDirMapped.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        try:
            windDirsVmapped.append(float(row[0]))
        except:
            x = 0

with open('./draftsMapped.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        try:
            draftsVmapped.append(float(row[0]))
        except:
            x = 0

with open('./blFlagsMapped.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        try:
            ballastLadenFlagsVmapped.append(row[0])
        except:
            x = 0

#################################
for dt in rrule(MINUTELY, dtstart=dtFrom, until=dtTo):
    dtTocompare = dt

    # ListfilteredDRFTaft=[i for i in range(0,len(draftAFT['data'])) if draftAFT['data'][i]['dt']==dtTocompare ]
    # filteredDRFTaft = "" if len(ListfilteredDRFTaft) == 0 else draftAFT['data'][ListfilteredDRFTaft[0]][ 'value' ]

    # ListfilteredDRFTfore = [ i for i in range(0, len(draftFORE[ 'data' ])) if draftFORE[ 'data' ][ i ][ 'dt' ] == dtTocompare ]
    # filteredDRFTfore = "" if len(ListfilteredDRFTfore) == 0 else draftFORE[ 'data' ][ ListfilteredDRFTfore[ 0 ] ]['value' ]

    ListfilteredDRFT = [i for i in range(0, len(draft['data'])) if draft['data'][i]['dt'] == dtTocompare]
    filteredDRFT = "" if len(ListfilteredDRFT) == 0 else draftAFT['data'][ListfilteredDRFT[0]]['value']

    ListfilteredWA = [i for i in range(0, len(windAngle['data'])) if windAngle['data'][i]['dt'] == dtTocompare]
    filteredWA = "" if len(ListfilteredWA) == 0 else windAngle["data"][ListfilteredWA[0]]['value']

    ListfilteredWS = [i for i in range(0, len(windSpeed['data'])) if windSpeed['data'][i]['dt'] == dtTocompare]
    filteredWS = "" if len(ListfilteredWS) == 0 else windSpeed["data"][ListfilteredWS[0]]['value']

    ListfilteredSO = [i for i in range(0, len(SpeedOvg['data'])) if SpeedOvg['data'][i]['dt'] == dtTocompare]
    filteredSO = "" if len(ListfilteredSO) == 0 else SpeedOvg["data"][ListfilteredSO[0]]['value']

    ListfilteredFO = [i for i in range(0, len(foc['data'])) if foc['data'][i]['dt'] == dtTocompare]
    filteredFO = "" if len(ListfilteredFO) == 0 else foc["data"][ListfilteredFO[0]]['value']

    ListfilteredRPM = [i for i in range(0, len(rpm['data'])) if rpm['data'][i]['dt'] == dtTocompare]
    filteredRPM = "" if len(ListfilteredRPM) == 0 else rpm["data"][ListfilteredRPM[0]]['value']

    ListfilteredPOW = [i for i in range(0, len(power['data'])) if power['data'][i]['dt'] == dtTocompare]
    filteredPOW = "" if len(ListfilteredPOW) == 0 else power["data"][ListfilteredPOW[0]]['value']
    try:
        data_writer.writerow([filteredDRFT, filteredWS, filteredWA, filteredSO, filteredRPM, filteredFO,
                              filteredPOW, str(dtTocompare)])

    except:
        x = 0

for s in neighboringTri:
    V1n = dataXnew[ s ][ 0 ]
    V2n = dataXnew[ s ][ 1 ]
    V3n = dataXnew[ s ][ 2 ]

    rpm1n = dataYnew[ s ][ 0 ]
    rpm2n = dataYnew[ s ][ 1 ]
    rpm3n = dataYnew[ s ][ 2 ]

    ###barycentric coords of neighboring points in relation to initial triangle
    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ V1n[ 0 ] - V3[ 0 ], V1n[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    W1n = solutions[ 0 ]
    W2n = solutions[ 1 ]
    W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

    B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
    nRpms.append(rpm1n - B1)

    nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
    ####################################

    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ V2n[ 0 ] - V3[ 0 ], V2n[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    W1n = solutions[ 0 ]
    W2n = solutions[ 1 ]
    W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

    B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
    nRpms.append(rpm2n - B1)

    nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
    ##################################################

    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ V3n[ 0 ] - V3[ 0 ], V3n[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    W1n = solutions[ 0 ]
    W2n = solutions[ 1 ]
    W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

    B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
    nRpms.append(rpm3n - B1)
    nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
    # ki = np.array([ V1n, V2n,
    # V3n ])
    # t5 = plt.Polygon(ki, fill=False, color='yellow', linewidth=3)
    # plt.gca().add_patch(t5)

    ##############################################
    distList = [ ]
    for k in range(0, len(triNew.vertices)):
        #################################################################################
        #########################################
        V1 = dataXnew[ triNew.vertices[ k ] ][ 0 ]
        V2 = dataXnew[ triNew.vertices[ k ] ][ 1 ]
        V3 = dataXnew[ triNew.vertices[ k ] ][ 2 ]

        rpm1 = dataYnew[ triNew.vertices[ k ] ][ 0 ]
        rpm2 = dataYnew[ triNew.vertices[ k ] ][ 1 ]
        rpm3 = dataYnew[ triNew.vertices[ k ] ][ 2 ]

        distV1 = math.sqrt(
            math.pow(candidatePoint[ 0 ] - V1[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V1[ 1 ], 2))
        distV2 = math.sqrt(
            math.pow(candidatePoint[ 0 ] - V2[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V2[ 1 ], 2))
        distV3 = math.sqrt(
            math.pow(candidatePoint[ 0 ] - V3[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V3[ 1 ], 2))

        distList.append(np.mean([ distV1, distV2, distV3 ]))

    minIndexDist = distList.index(np.min(distList))
    k = minIndexDist
    neighboringVertices1 = [ ]
    #####################################
    for u in range(0, 2):
        try:
            neighboringVertices1.append(triNew.vertex_neighbor_vertices[ 1 ][
                                        triNew.vertex_neighbor_vertices[ 0 ][
                                            triNew.vertices[ k ][ u ] ]:
                                        triNew.vertex_neighbor_vertices[ 0 ][
                                            triNew.vertices[ k ][ u ] + 1 ] ])
        except:
            break
        neighboringTri = triNew.vertices[
            triNew.find_simplex(dataXnew[ np.concatenate(np.array(neighboringVertices1)) ]) ]
        # simplex = triNew.vertices[ triNew.find_simplex(
        # [ dataXnew[ triNew.vertices[ k ] ][ 0 ], dataXnew[ triNew.vertices[ k ] ][ 1 ],
        # dataXnew[ triNew.vertices[ k ] ][ 2 ] ]) ]

    for s in neighboringTri:
        V1 = dataXnew[ s ][ 0 ]
        V2 = dataXnew[ s ][ 1 ]
        V3 = dataXnew[ s ][ 2 ]

        rpm1 = dataYnew[ s ][ 0 ]
        rpm2 = dataYnew[ s ][ 1 ]
        rpm3 = dataYnew[ s ][ 2 ]

        distV1 = math.sqrt(
            math.pow(candidatePoint[ 0 ] - V1[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V1[ 1 ], 2))
        distV2 = math.sqrt(
            math.pow(candidatePoint[ 0 ] - V2[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V2[ 1 ], 2))
        distV3 = math.sqrt(
            math.pow(candidatePoint[ 0 ] - V3[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V3[ 1 ], 2))

        W1x = 1 / distV1 if distV1 != 0 else 0
        W2x = 1 / distV2 if distV2 != 0 else 0
        W3x = 1 / distV3 if distV3 != 0 else 0

        ##Barycentric coordinates

        W1 = (V2[ 1 ] - V3[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (
                candidatePoint[ 1 ] - V3[ 1 ]) \
             / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

        W2 = (V3[ 1 ] - V1[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V1[ 0 ] - V3[ 0 ]) * (
                candidatePoint[ 1 ] - V3[ 1 ]) \
             / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

        W3 = 1 - W1 - W2
        ##############################################################
        prediction = (W1 * rpm1 + W2 * rpm2 + W3 * rpm3) / (W1 + W2 + W3)
        pred1S = (W1x * rpm1 + W2x * rpm2 + W3x * rpm3) / (W1x + W2x + W3x)
        preds.append(pred1S)
    #################################################################################
    #########################################
    V1 = dataXnew[ triNew.vertices[ k ] ][ 0 ]
    V2 = dataXnew[ triNew.vertices[ k ] ][ 1 ]
    V3 = dataXnew[ triNew.vertices[ k ] ][ 2 ]

    rpm1 = dataYnew[ triNew.vertices[ k ] ][ 0 ]
    rpm2 = dataYnew[ triNew.vertices[ k ] ][ 1 ]
    rpm3 = dataYnew[ triNew.vertices[ k ] ][ 2 ]

    distV1 = math.sqrt(
        math.pow(candidatePoint[ 0 ] - V1[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V1[ 1 ], 2))
    distV2 = math.sqrt(
        math.pow(candidatePoint[ 0 ] - V2[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V2[ 1 ], 2))
    distV3 = math.sqrt(
        math.pow(candidatePoint[ 0 ] - V3[ 0 ], 2) + math.pow(candidatePoint[ 1 ] - V3[ 1 ], 2))

    W1x = 1 / distV1 if distV1 != 0 else 0
    W2x = 1 / distV2 if distV2 != 0 else 0
    W3x = 1 / distV3 if distV3 != 0 else 0

    ##Barycentric coordinates

    W1 = (V2[ 1 ] - V3[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (
            candidatePoint[ 1 ] - V3[ 1 ]) \
         / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

    W2 = (V3[ 1 ] - V1[ 1 ]) * (candidatePoint[ 0 ] - V3[ 0 ]) + (V1[ 0 ] - V3[ 0 ]) * (
            candidatePoint[ 1 ] - V3[ 1 ]) \
         / (V2[ 1 ] - V3[ 1 ]) * (V1[ 0 ] - V3[ 0 ]) + (V3[ 0 ] - V2[ 0 ]) * (V1[ 1 ] - V3[ 1 ])

    W3 = 1 - W1 - W2
    #############################

    prediction = (W1 * rpm1 + W2 * rpm2 + W3 * rpm3) / (W1 + W2 + W3)
    pred1 = (W1x * rpm1 + W2x * rpm2 + W3x * rpm3) / (W1x + W2x + W3x)
    # preds.append(prediction)
    preds.append(pred1)
    LinearPred = W1 * rpm1 + W2 * rpm2 + W3 * rpm3
    pred = (math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3)
    #####################################################################
    flgR = False
    flgV = False
    ##efaptomena trigwna panw sto trigwno sto opoio anikei to Point estimate
    tri1 = np.array(triNew.vertices[
                        [ i for i, x in enumerate(list(triNew.vertices[ :, 0 ] == triNew.vertices[ k, 0 ]))
                          if x ] ])
    # tri1 = np.delete(tri1,triNew.vertices[k])

    tri11 = [ ]
    k_set = set(triNew.vertices[ k ])

    for tr in tri1:
        a_set = set(tr)
        tpm1 = dataXnew[ tr[ 0 ] ]
        tpm2 = dataXnew[ tr[ 1 ] ]
        tpm3 = dataXnew[ tr[ 2 ] ]
        if len(k_set.intersection(a_set)) == 2 and \
                (tr != triNew.vertices[ k ]).any():
            tri11.append(tr)

    tri2 = np.array(triNew.vertices[
                        [ i for i, x in enumerate(list(triNew.vertices[ :, 1 ] == triNew.vertices[ k, 1 ]))
                          if x ] ])
    # tri2 = np.delete(tri2, triNew.vertices[ k ])
    tri22 = [ ]
    tri11 = np.array(tri11)
    for tr in tri2:
        b_set = set(tr)
        tpm1 = dataXnew[ tr[ 0 ] ]
        tpm2 = dataXnew[ tr[ 1 ] ]
        tpm3 = dataXnew[ tr[ 2 ] ]
        if len(k_set.intersection(b_set)) == 2 \
                and \
                (tr != triNew.vertices[ k ]).any():
            tri22.append(tr)

    tri3 = np.array(triNew.vertices[
                        [ i for i, x in enumerate(list(triNew.vertices[ :, 2 ] == triNew.vertices[ k, 2 ]))
                          if x ] ])
    # tri3 = np.delete(tri3, triNew.vertices[ k ])
    tri33 = [ ]
    tri22 = np.array(tri22)
    for tr in tri3:
        c_set = set(tr)
        tpm1 = dataXnew[ tr[ 0 ] ]
        tpm2 = dataXnew[ tr[ 1 ] ]
        tpm3 = dataXnew[ tr[ 2 ] ]
        if len(k_set.intersection(c_set)) == 2 \
                and \
                (tr != triNew.vertices[ k ]).any():
            tri33.append(tr)

    tri33 = np.array(tri33)
    #####

    ######
    # attched_vertex = np.concatenate([tri11,tri22,tri33])

    rpms = [ ]
    vs = [ ]
    if tri11 != [ ]:
        for tr in tri11:
            vi = [ x for x in tr if x not in triNew.vertices[ k ] ]
            rpms.append(dataYnew[ vi ])
            vs.append(dataXnew[ vi ])

    if tri22 != [ ]:
        for tr in tri22:
            vi = [ x for x in tr if x not in triNew.vertices[ k ] ]
            rpms.append(dataYnew[ vi ])
            vs.append(dataXnew[ vi ])

    if tri33 != [ ]:
        for tr in tri33:
            vi = [ x for x in tr if x not in triNew.vertices[ k ] ]
            rpms.append(dataYnew[ vi ])
            vs.append(dataXnew[ vi ])

    if len(rpms) < 3:
        flgR = True
        rpms.append(np.array([ 0 ]))
    try:
        rpm12 = rpms[ 0 ]
        rpm23 = rpms[ 1 ]
        rpm31 = rpms[ 2 ]
    except:
        x = 0

    if len(vs) < 3:
        flgV = True
        vs.append([ np.array([ 0, 0 ]) ])

    v12 = vs[ 0 ][ 0 ]
    v23 = vs[ 1 ][ 0 ]
    v31 = vs[ 2 ][ 0 ]

    #################################################################################33
    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ candidatePoint[ 0 ] - V3[ 0 ], candidatePoint[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    W1 = solutions[ 0 ]
    W2 = solutions[ 1 ]
    W3 = 1 - solutions[ 0 ] - solutions[ 1 ]

    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ V1[ 0 ] - V3[ 0 ], V1[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    ###################################################
    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ v12[ 0 ] - V3[ 0 ], v12[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    W12_1 = solutions[ 0 ]
    W12_2 = solutions[ 1 ]
    W12_3 = 1 - solutions[ 0 ] - solutions[ 1 ]

    cond_numer1 = np.linalg.cond(eq1)
    ###########################

    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ v23[ 0 ] - V3[ 0 ], v23[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    W23_1 = solutions[ 0 ]
    W23_2 = solutions[ 1 ]
    W23_3 = 1 - solutions[ 0 ] - solutions[ 1 ]
    cond_numer2 = np.linalg.cond(eq1)
    ####################################################

    eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                     [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

    eq2 = np.array([ v31[ 0 ] - V3[ 0 ], v31[ 1 ] - V3[ 1 ] ])
    solutions = np.linalg.solve(eq1, eq2)

    W31_1 = solutions[ 0 ]
    W31_2 = solutions[ 1 ]
    W31_3 = 1 - solutions[ 0 ] - solutions[ 1 ]
    cond_numer3 = np.linalg.cond(eq1)
    #######################################################
    # efaptomena

    #######################################33
    xPredOUT = ((math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) + \
                2 * W1 * W2 * rpm12 + 2 * W2 * W3 * rpm23 + 2 * W1 * W3 * (
                        rpm1 + rpm2 + rpm3 + rpm23 + rpm12) / 5)
    ###################################################################
    nRpms = [ ]
    nGammas = [ ]
    rpm31 = rpm31 if rpm31 != 0 else (rpm1 + rpm2 + rpm3 + rpm23 + rpm12) / 5

    B1 = (math.pow(W12_1, 2) * rpm1 + math.pow(W12_2, 2) * rpm2 + math.pow(W12_3, 2) * rpm3)
    nRpms.append(rpm12[ 0 ] - B1)

    B1 = (math.pow(W23_1, 2) * rpm1 + math.pow(W23_2, 2) * rpm2 + math.pow(W23_3, 2) * rpm3)
    nRpms.append(rpm23[ 0 ] - B1)

    B1 = (math.pow(W31_1, 2) * rpm1 + math.pow(W31_2, 2) * rpm2 + math.pow(W31_3, 2) * rpm3)
    nRpms.append(rpm31[ 0 ] - B1)

    nGammas.append(np.array([ 2 * W12_1 * W12_2, 2 * W12_1 * W12_3, 2 * W12_2 * W12_3 ]))
    nGammas.append(np.array([ 2 * W23_1 * W23_2, 2 * W23_1 * W23_3, 2 * W23_2 * W23_3 ]))
    nGammas.append(np.array([ 2 * W31_1 * W31_2, 2 * W31_1 * W31_3, 2 * W31_2 * W31_3 ]))

    for s in neighboringTri:
        V1n = dataXnew[ s ][ 0 ]
        V2n = dataXnew[ s ][ 1 ]
        V3n = dataXnew[ s ][ 2 ]

        rpm1n = dataYnew[ s ][ 0 ]
        rpm2n = dataYnew[ s ][ 1 ]
        rpm3n = dataYnew[ s ][ 2 ]

        ###barycentric coords of neighboring points in relation to initial triangle
        eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                         [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

        eq2 = np.array([ V1n[ 0 ] - V3[ 0 ], V1n[ 1 ] - V3[ 1 ] ])
        solutions = np.linalg.solve(eq1, eq2)

        W1n = solutions[ 0 ]
        W2n = solutions[ 1 ]
        W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

        B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
        nRpms.append(rpm1n - B1)

        nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
        ####################################

        eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                         [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

        eq2 = np.array([ V2n[ 0 ] - V3[ 0 ], V2n[ 1 ] - V3[ 1 ] ])
        solutions = np.linalg.solve(eq1, eq2)

        W1n = solutions[ 0 ]
        W2n = solutions[ 1 ]
        W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

        B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
        nRpms.append(rpm2n - B1)

        nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
        ##################################################

        eq1 = np.array([ [ (V1[ 0 ] - V3[ 0 ]), (V2[ 0 ] - V3[ 0 ]) ],
                         [ (V1[ 1 ] - V3[ 1 ]), (V2[ 1 ] - V3[ 1 ]) ] ])

        eq2 = np.array([ V3n[ 0 ] - V3[ 0 ], V3n[ 1 ] - V3[ 1 ] ])
        solutions = np.linalg.solve(eq1, eq2)

        W1n = solutions[ 0 ]
        W2n = solutions[ 1 ]
        W3n = 1 - solutions[ 0 ] - solutions[ 1 ]

        B1 = (math.pow(W1n, 2) * rpm1 + math.pow(W2n, 2) * rpm2 + math.pow(W3n, 2) * rpm3)
        nRpms.append(rpm3n - B1)
        nGammas.append(np.array([ 2 * W1n * W2n, 2 * W1n * W3n, 2 * W2n * W3n ]))
        # ki = np.array([ V1n, V2n,
        # V3n ])
        # t5 = plt.Polygon(ki, fill=False, color='yellow', linewidth=3)
        # plt.gca().add_patch(t5)
    # plt.show()
    ####solve least squares opt. problem
    # nRPms : y [1xn matrix]
    # nGammas : x [3xn matrix]

    nGammas = np.array(nGammas)
    nRpms = np.array(nRpms)
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    sr = sp.Earth()
    splApprx = sr.fit(nGammas.reshape(-1, 3), nRpms.reshape(-1, 1))
    leastSqApprx = lr.fit(nGammas.reshape(-1, 3), nRpms.reshape(-1, 1))
    XpredN = ((math.pow(W2, 2) * rpm2 + math.pow(W1, 2) * rpm1 + math.pow(W3, 2) * rpm3) +
              2 * W1 * W2 * leastSqApprx.coef_[ 0 ][ 0 ] +
              2 * W2 * W3 * leastSqApprx.coef_[ 0 ][ 1 ] +
              2 * W1 * W3 * leastSqApprx.coef_[ 0 ][ 2 ])
    x = 1