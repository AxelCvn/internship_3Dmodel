import numpy as np
import random
from sklearn import random_projection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import neural_network
import csv
import os
import time
import ast
import matplotlib.pyplot as plt

##############################################################################################################
def splitdata(fcFileT, rotFileT, fcFileVal, rotFileVal):
    featList = []
    listTime = time.time()
    with open(fcFileT) as allFc :
        for line in allFc :
            featList.append(ast.literal_eval(line))
    print("--- %s seconds --- to prepare data" % (time.time() - listTime))

    rotList = []
    with open(rotFileT) as rot_file :
        rotList = rot_file.read().split(',')
    for i in range(len(rotList)) :
        rotList[i] = float(rotList[i])

    fcSet = []
    rotSet = []
    index_shuf = range(len(featList))
    print min(index_shuf)
    random.shuffle(index_shuf)
    np.savetxt('shuffle.csv', index_shuf, delimiter=",")
    for i in index_shuf:
        fcSet.append(featList[i])
        rotSet.append(rotList[i])

    val_data = []
    with open(fcFileVal) as allFcVal :
        for line in allFcVal :
            val_data.append(ast.literal_eval(line))

    val_label = []
    with open(rotFileVal) as rot_fileVal :
        val_label = rot_fileVal.read().split(',')
    for i in range(len(val_label)) :
        val_label[i] = float(val_label[i])

    print 'OK'
    dataSize = len(featList)
    print " LENGTH DATA = " + str(dataSize)

    trainSize = dataSize*70//100
    testSize = dataSize - trainSize

    training_data = fcSet[:trainSize]
    training_label = rotSet[:trainSize]

    test_data = fcSet[-testSize:]
    print len(test_data)

    test_label = rotSet[-testSize:]
    np.savetxt('shuffledTest.csv', test_label, delimiter=",")
    testItems = index_shuf[-testSize:]

    print "Data is ready for machine learning"
    return training_data, training_label, test_data, test_label, val_data, val_label, index_shuf

##############################################################################################################


def all(trainData,trainLabel, valData, valLabel, nbRun):
    cwd = os.getcwd()
    print cwd
    csvDir =  cwd + '/predict/'
    if not os.path.exists(csvDir):
        os.mkdir(csvDir)
    scoreDir = cwd + '/score/'
    if not os.path.exists(scoreDir):
        os.mkdir(scoreDir)

    first_time = time.time()

    rfr_res = []
    rfr_val = []

    svr_res_lin = []
    svr_val_lin = []

    svr_res_poly = []
    svr_val_poly = []

    nnr_res = []
    nnr_val = []


    for z in range (nbRun):
        print 'Run number :' + str(nbRun)
        inputDataTrain, inputMetaDataTrain, inputDataTest, inputMetaDataTest, inputDataVal, inputMetaDataVal, index = splitdata(trainData,trainLabel, valData, valLabel)

        size_test = len(inputMetaDataTest)
        size_val = len(inputMetaDataVal)

        ###################### RANDOM FOREST REGRESSOR ######################
        errValRFR = []
        rfr_iter = []
        rfr_iter_val = []
        x = [10, 50, 100]
        for j in x :
            forestTime = time.time()
            nb_estim = j
            # RandomForestRegressor
            rgs = RandomForestRegressor(n_estimators=nb_estim)
            rgs = rgs.fit(inputDataTrain, inputMetaDataTrain)
            #pred = rgs.predict(inputDataTest)

            resTrain = rgs.score(inputDataTrain, inputMetaDataTrain)
            print "TRAIN results with RandomForestRegressor : " + str(resTrain)


            print "estimators :" + str(nb_estim)
            res = rgs.score(inputDataTest, inputMetaDataTest)
            rfr_iter.append(res)
            print "TEST results with RandomForestRegressor : " + str(res)

            rfr_test_pred_path = csvDir + 'rfr_pred_test_'+str(z)+'_'+str(j)+'.csv'
            with open(rfr_test_pred_path, 'wb') as csvfile:
                filewriter = csv.writer(csvfile)
                rfrPred = rgs.predict(inputDataTest)
                filewriter.writerow(rfrPred)

            val = rgs.score(inputDataVal, inputMetaDataVal)
            rfr_iter_val.append(val)
            print "VALIDATION results with RandomForestRegressor : " + str(val)

            ############ MAKE AND SAVE PREDICTION ############
            rfr_val_pred_path = csvDir + 'rfr_pred_val_'+str(z)+'_'+str(j)+'.csv'
            with open(rfr_val_pred_path, 'wb') as csvfile:
                filewriter = csv.writer(csvfile)
                valPred = rgs.predict(inputDataVal)
                #Compute average error in prediction
                errVal = sum(abs(valPred - inputMetaDataVal))/1000
                errValRFR.append(errVal)
                filewriter.writerow(valPred)

            print("--- %s seconds --- Estimation Time" % (time.time() - forestTime))

        print 'ERROR AVERAGE FOR x = [10, 50, 100] : ' + str(errValRFR)

        rfr_res.append(rfr_iter)
        rfr_val.append(rfr_iter_val)

        #####################################################################


        ##################### SUPPORT VECTOR REGRESSION #####################

        errValSVR = []

        #svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr_lin = svm.SVR(kernel='linear', C=1e3)
        svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2)

        #y_rbf = svr_rbf.fit(inputDataTest, inputMetaDataTest)
        y_lin = svr_lin.fit(inputDataTrain, inputMetaDataTrain)
        y_poly = svr_poly.fit(inputDataTrain, inputMetaDataTrain)

        ################## TRAINING DATA ##################
        svr_linTrain = y_lin.score(inputDataTrain, inputMetaDataTrain)
        svr_polyTrain = y_poly.score(inputDataTrain, inputMetaDataTrain)
        print "rbs_linTrain = " + str(svr_linTrain)
        print "rbs_polyTrain = " + str(svr_polyTrain)


        #################### TEST DATA ####################
        #LINEAR SVR
        svr_lin = y_lin.score(inputDataTest, inputMetaDataTest)
        print "rbs_linTest = " + str(svr_lin)
        svr_res_lin.append(svr_lin)

        svr_lin_test_pred_path = csvDir + 'svr_lin_pred_test_'+str(z)+'.csv'
        with open(svr_lin_test_pred_path, 'wb') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(y_lin.predict(inputDataTest))

        #POLY SVR
        svr_poly = y_poly.score(inputDataTest, inputMetaDataTest)
        print "rbf_polyTest = " + str(svr_poly)
        svr_res_poly.append(svr_poly)

        ############ MAKE AND SAVE PREDICTION ############
        svr_poly_test_pred_path = csvDir + 'svr_poly_pred_test_'+str(z)+'.csv'
        with open(svr_poly_test_pred_path, 'wb') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(y_poly.predict(inputDataTest))


        ################# VALIDATION DATA ##################
        #LINEAR SVR
        svr_lin_val = y_lin.score(inputDataVal, inputMetaDataVal)
        print "rbs_lin_val = " + str(svr_lin_val)
        svr_val_lin.append(svr_lin_val)

        ############ MAKE AND SAVE PREDICTION ############
        svr_lin_val_pred_path = csvDir + 'svr_lin_pred_val_'+str(z)+'.csv'
        with open(svr_lin_val_pred_path, 'wb') as csvfile:
            filewriter = csv.writer(csvfile)
            valPred = y_lin.predict(inputDataVal)
            #Compute average error in prediction
            errVal = sum(abs(valPred - inputMetaDataVal))/1000
            errValSVR.append(errVal)
            filewriter.writerow(valPred)

        #POLY SVR
        svr_poly_val = y_poly.score(inputDataVal, inputMetaDataVal)
        print "svr_poly_val = " + str(svr_poly_val)
        svr_val_poly.append(svr_poly_val)

        ############ MAKE AND SAVE PREDICTION ############
        svr_poly_val_pred_path = csvDir + 'svr_poly_pred_val_'+str(z)+'.csv'
        with open(svr_poly_val_pred_path, 'wb') as csvfile:
            filewriter = csv.writer(csvfile)
            valPred = y_poly.predict(inputDataVal)
            #Compute average error in prediction
            errVal = sum(abs(valPred - inputMetaDataVal))/1000
            errValSVR.append(errVal)
            filewriter.writerow(valPred)



        ##################### NEURAL NETWORK REGRESSION #####################
        hd_lrs = [100,200,300]
        errValNNR = []

        nnr_iter = []
        nnr_iter_val = []
        for h in hd_lrs :
            NNRTime = time.time()
            nnr = neural_network.MLPRegressor(hidden_layer_sizes=h,activation='identity',solver='adam')
            nnr = nnr.fit(inputDataTrain, inputMetaDataTrain)
            resVal = nnr.score(inputDataTrain, inputMetaDataTrain)

            print "TRAIN results with NeuralNetworkRegressor : " + str(resVal)
            res = nnr.score(inputDataTest, inputMetaDataTest)

            nnr_test_pred_path = csvDir + 'nnr_pred_test_'+str(z)+'_'+str(h)+'.csv'
            with open(nnr_test_pred_path, 'wb') as csvfile:
                filewriter = csv.writer(csvfile)
                filewriter.writerow(nnr.predict(inputDataTest))

            print "TEST results with NeuralNetworkRegressor : " + str(res)
            nnr_iter.append(res)

            val = nnr.score(inputDataVal, inputMetaDataVal)
            print "VALIDATION results with NeuralNetworkRegressor : " + str(val)
            nnr_iter_val.append(val)

            ############ MAKE AND SAVE PREDICTION ############
            nnr_val_pred_path = csvDir + 'nnr_pred_val_'+str(z)+'_'+str(h)+'.csv'
            with open(nnr_val_pred_path, 'wb') as csvfile:
                filewriter = csv.writer(csvfile)
                valPred = nnr.predict(inputDataVal)
                #Compute average error in prediction
                errVal = sum(abs(valPred - inputMetaDataVal))/1000
                errValNNR.append(errVal)
                filewriter.writerow(valPred)

            print("--- %s seconds --- NNRTime Time" % (time.time() - NNRTime))
        nnr_res.append(nnr_iter)
        nnr_val.append(nnr_iter_val)


    ################ SAVING SCORE ################
    rfr_res=np.asarray(rfr_res)
    np.savetxt('rfr_ran.csv', rfr_res, delimiter=",")

    svr_res_lin=np.asarray(svr_res_lin)
    np.savetxt('svr_lin_ran.csv', svr_res_lin, delimiter=",")
    svr_res_poly=np.asarray(svr_res_poly)
    np.savetxt('svr_poly_ran.csv', svr_res_poly, delimiter=",")

    nnr_res=np.asarray(nnr_res)
    np.savetxt('nnr_ran.csv', nnr_res, delimiter=",")

    rfr_val=np.asarray(rfr_val)
    np.savetxt('rfr_ran_val.csv', rfr_val, delimiter=",")

    svr_val_lin=np.asarray(svr_val_lin)
    np.savetxt('svr_lin_ran_val.csv', svr_val_lin, delimiter=",")
    svr_val_poly=np.asarray(svr_val_poly)
    np.savetxt('svr_poly_ran_val.csv', svr_val_poly, delimiter=",")

    nnr_val=np.asarray(nnr_val)
    np.savetxt('nnr_ran_val.csv', nnr_val, delimiter=",")

    print("--- %s seconds --- TOTAL TIME" % (time.time() - first_time))
    return rfr_val_pred_path
    #####################################################################
