import numpy as np
from datetime import datetime
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import pickle
import math
import time
import os
from snaptime_helper import FillData
import pandas as pd

def _test_LSTM(input_directory,model,test_indices,sum,sumsq,obj,interval,X_cols,timeslice,granularity,train_indices=None):
    files = os.listdir(input_directory)
    y_true_test = []
    y_pred_test = []
    y_true_train = []
    y_pred_train = []
    files = [file for file in os.listdir(input_directory)]
    epoch = datetime.utcfromtimestamp(0)
    
    for i in xrange(len(files)):
        print "Testing file", i
        file = files[i]
        init_epoch = long((datetime.strptime('_'.join(file.split('_')[:2]),'%Y%m%d_%H') - epoch).total_seconds()*1000)
        # print time.time() - time1
        data = None
        time1 = time.time()
        if file in test_indices or file in train_indices:
            data = obj.createAndFillData(os.path.join(input_directory,file),init_epoch,timeslice,granularity)
            data = data[:, X_cols]
            for j in xrange(data.shape[1]):
                if sumsq[j] != 0:
                    data[:,j] = (data[:,j]-sum[j])/sumsq[j]


        else:
            continue

        time2 = time.time()
        print 'PHASE 1 took: ', time2-time1

        if file in test_indices:
            # data = pd.read_csv(os.path.join(input_directory,file),sep='\t',header=None,low_memory=False)[X_cols].as_matrix()
            time175 = time.time()
            # print time.time() - time15
            # print time.time() - time175
            positives = test_indices[file]['positive']
            negatives = test_indices[file]['negative']
            time2 = time.time()
            # print "Phase 1 took", time2-time1
            if len(positives) > 0 and len(negatives) > 0:
                X_test = np.concatenate(([data[idx:idx+interval,:] for idx in positives],[data[idx:idx+interval,:] for idx in negatives]))
            elif len(positives) > 0:
                X_test = np.array([data[idx:idx+interval,:] for idx in positives])
            else:
                X_test = np.array([data[idx:idx+interval,:] for idx in negatives])
            for idx in positives:
                y_true_test.append([0,1])
            for idx in negatives:
                y_true_test.append([1,0])
            time3 = time.time()
            print "Phase 2 took", time3 - time2
            y_pred_test += model.predict(X_test).tolist()
            print "Phase 3 took", time.time() - time3

        time4 = time.time()


        if train_indices != None:
            if file in train_indices:
                # data = obj.createAndFillData(os.path.join(input_directory,file),init_epoch,timeslice,granularity)[:,X_cols]
                # data = pd.read_csv(os.path.join(input_directory,file),sep='\t',header=None,low_memory=False)[X_cols].as_matrix()
                # for j in xrange(data.shape[1]):
                #     if sumsq[j] != 0:
                #         data[:,j] = (data[:,j]-sum[j])/sumsq[j]
                positives = train_indices[file]['positive']
                negatives = train_indices[file]['negative']
                if len(positives) > 0 and len(negatives) > 0:
                    X_test = np.concatenate(([data[idx:idx+interval,:] for idx in positives],[data[idx:idx+interval,:] for idx in negatives]))
                elif len(positives) > 0:
                    X_test = np.array([data[idx:idx+interval,:] for idx in positives])
                else:
                    X_test = np.array([data[idx:idx+interval,:] for idx in negatives])
                for idx in positives:
                    y_true_train.append([0,1])
                for idx in negatives:
                    y_true_train.append([1,0])
                y_pred_train += model.predict(X_test).tolist()

                print "Phase 4 took:", time.time() - time4
                print "Total Runtime was", time.time()- time1


    fpr,tpr,thres = roc_curve([y_true_test[i][1] for i in xrange(len(y_true_test))],[y_pred_test[i][1] for i in xrange(len(y_pred_test))])
    test_score =  roc_auc_score([y_true_test[i][1] for i in xrange(len(y_true_test))],[y_pred_test[i][1] for i in xrange(len(y_pred_test))])
    if train_indices != None:
        fpr_t,tpr_t,thres_t = roc_curve([y_true_train[i][1] for i in xrange(len(y_true_train))],[y_pred_train[i][1] for i in xrange(len(y_pred_train))])
        train_score = roc_auc_score([y_true_train[i][1] for i in xrange(len(y_true_train))],[y_pred_train[i][1] for i in xrange(len(y_pred_train))])
    else:
        fpr_t,tpr_t,thres_t,train_score=None,None,None,None,None
    return fpr,tpr,thres,test_score,fpr_t,tpr_t,thres_t,train_score

def _train_LSTM(input_directory,train_indices,buffering,sum,sumsq,LSTM_model,iterations,obj,interval,X_cols,timeslice,granularity, batches):
    global_X = []
    global_Y = []
    global_idx = 36000000
    global_file_idx = -1
    files = [file for file in os.listdir(input_directory)]
    epoch = datetime.utcfromtimestamp(0)
    def train_generator():
        # print 'in generator'
        global_file_idx = -1
        global_idx = 36000000
        global_X = []
        global_Y = []
        while True:
            # print 'in loops'
            #refresh with another file's data

            if global_idx >= len(global_X):
                global_idx = 0
                global_file_idx += 1 
                global_file_idx = global_file_idx % len(files)

                while files[global_file_idx] not in train_indices:
            		global_file_idx += 1 
            		global_file_idx = global_file_idx % len(files)
            		
                global_X = []
                global_Y = []
                init_epoch = long((datetime.strptime('_'.join(files[global_file_idx].split('_')[:2]),'%Y%m%d_%H') - epoch).total_seconds()*1000)
                positives = train_indices[files[global_file_idx]]['positive']
                negatives = train_indices[files[global_file_idx]]['negative']
                data = obj.createAndFillData(os.path.join(input_directory,files[global_file_idx]),init_epoch,timeslice,granularity)
                data = data[:, X_cols]
                # data = pd.read_csv(os.path.join(input_directory,file),sep='\t',header=None,low_memory=False)[X_cols].as_matrix()
                if len(positives) > 0 and len(negatives) > 0:
                    global_Y = np.concatenate(([[0,1] for idx in positives],[[1,0] for idx in negatives]))
                    global_X = np.concatenate(([data[idx:idx+interval,:] for idx in positives],[data[idx:idx+interval,:] for idx in negatives]))
                elif len(positives) > 0:
                    global_Y = np.array([[0,1] for idx in positives])
                    global_X = np.array([data[idx:idx+interval,:] for idx in positives])
                elif len(negatives) > 0:
                    global_Y = np.array([[1,0] for idx in negatives])
                    global_X = np.array([data[idx:idx+interval,:] for idx in negatives])
                order = np.arange(len(global_X))
                np.random.shuffle(order)
                global_X = global_X[order]
                for j in xrange(global_X.shape[2]):
                    if sumsq[j] != 0:
                        global_X[:,:,j] = (global_X[:,:,j] - sum[j])/sumsq[j]
                global_Y = global_Y[order]
            val = (global_X[global_idx:min(global_idx+buffering,len(global_X))],global_Y[global_idx:min(global_idx+buffering,len(global_Y))])
            global_idx = global_idx + buffering
            yield val

    print 'Fitting Generator'
    history = LSTM_model.fit_generator(train_generator(),batches,iterations)
    pickle.dump(history.history, open('history.pkl', 'wb'))
    return LSTM_model

def run_LSTM(input_directory,X_cols,Y_cols,interval,lookahead,timeslice,granularity,y_differentiator,buffering,LSTM_model,iterations,imbalance=10,lastval=True, testAll = True):
    """input directory - directory containing files in snaptime format
    X_cols - columns with independent data
    Y_cols - columns with dependent data
    interval - window size
    lookahead - time duration for prediction
    timeslice - total length of timeseries in milliseconds
    granularity - minimum difference between timestamps in milliseconds
    y_differentiator : lambda function for specifying whether a certain y value maps to 1
    buffering : LSTM batch size
    LSTM_model : deep learning model
    iterations : number of training iterations
    imbalance : negative/positive imbalance ratio used in training
    lastval - if True, consider only the last point of the lookahead window for generating an example

    returns:
    fpr - test false positive rate array
    tpr - test true positive rate array
    thres - test roc thresholds array
    test_score - test auc score
    fpr_t - train false positive rate array
    tpr_t - train true positive rate array
    thres_t- train roc threshold array
    train_score - train auc score
    """
    #prepare training and test indices and preprocess the data
    data_train = {}
    data_test = {}
    sum = [0 for i in xrange(len(X_cols))]
    sumsq = [0 for i in xrange(len(X_cols))]
    files = os.listdir(input_directory)
    epoch = datetime.utcfromtimestamp(0)
    obj = FillData()
    total = 0
    xxx = 0
    batches = 0
    for file in files:
        print xxx, ' out of ', len(files)
        xxx += 1
        init_epoch = long((datetime.strptime('_'.join(file.split('_')[:2]),'%Y%m%d_%H') - epoch).total_seconds()*1000)
        # data = pd.read_csv(os.path.join(input_directory,file),sep='\t',header=None,low_memory=False)[X_cols].as_matrix()
        data = obj.createAndFillData(os.path.join(input_directory,file),init_epoch,timeslice,granularity)
        # for i, element in enumerate(data[0]):
        #     print (i + 1), element
        # print type(data)
        # print data.shape
        full_data_Y = data[:,Y_cols]
        full_data_X = data[:,X_cols]
        total += len(full_data_X)
        #preprocess the data
        for j in xrange(full_data_X.shape[1]):
            sum[j] += np.sum(full_data_X[:,j])
        for j in xrange(full_data_X.shape[1]):
            sumsq[j] += np.sum(full_data_X[:,j]*full_data_X[:,j])
        i = 0
        counter = 0
        pos_idx = []
        neg_idx = []
        while i + interval+lookahead - 1 < len(full_data_Y):
            pointer = 1
            while pointer <= lookahead and y_differentiator(full_data_Y[i+interval-1+pointer]) == False:
                pointer += 1
            if pointer <= lookahead:
                for j in xrange(i,i+pointer):
                    if lastval == True and np.any(map(y_differentiator,full_data_Y[j:j+interval,:])):
                        continue
                    pos_idx.append(j)
                i += pointer+interval
                while i < len(full_data_Y) and y_differentiator(full_data_Y[i]) == True:
                    i = i + 1
            else:
                neg_idx.append(i)
                i = i + 1
        pos_idx = np.array(pos_idx)
        if len(pos_idx) == 0:
            continue
        temp_neg_idx = np.array(neg_idx)
        samples = np.arange(len(temp_neg_idx)/(lookahead*10))
        np.random.shuffle(samples)
        neg_idx = []
        for i in samples:
            neg_idx += temp_neg_idx[lookahead*10*i:lookahead*10*(i+1)].tolist()
        neg_idx = np.array(neg_idx)
        np.random.shuffle(neg_idx)
        temp_neg_idx = neg_idx[:imbalance*len(pos_idx)]
        start = imbalance*len(pos_idx)
        np.random.shuffle(pos_idx)
        # print 'Checking here'
        # print type(temp_neg_idx)
        # print temp_neg_idx.shape
        # print pos_idx.shape
        # print temp_neg_idx
        data_train[file] ={'positive':pos_idx[:int(0.9*len(pos_idx))],'negative':temp_neg_idx[:int(0.9*len(temp_neg_idx))]}
        batches += math.ceil((len(data_train[file]['positive']) + len(data_train[file]['negative'])) / float(buffering))
        if not testAll:
            data_test[file] ={'positive':pos_idx[int(0.9*len(pos_idx)):],'negative':temp_neg_idx[int(0.9*len(temp_neg_idx)):]}
        else:
            data_test[file] ={'positive':pos_idx[int(0.9*len(pos_idx)):],'negative':np.concatenate((temp_neg_idx[int(0.9*len(temp_neg_idx)):],neg_idx[start:]))}
    sum,sumsq = np.array(sum),np.array(sumsq)
    sum /= total
    sumsq = (sumsq/total - sum**2)**0.5
    pickle.dump((sum, sumsq), open('sums.pkl', 'wb'))
    pickle.dump((data_train, data_test), open('data.pkl', 'wb'))
    pickle.dump(batches, open('batches.pkl', 'wb'))
    #train and test
    print 'Reaching training'
    model = _train_LSTM(input_directory,data_train,buffering,sum,sumsq,LSTM_model,iterations,obj,interval,X_cols,timeslice,granularity, batches)
    model.save_weights('model.h5')
    fpr,tpr,thres,test_score,fpr_t,tpr_t,thres_t,train_score = _test_LSTM(input_directory,model,data_test,sum,sumsq,obj,interval,X_cols,timeslice,granularity,data_train)
    return fpr,tpr,thres,test_score,fpr_t,tpr_t,thres_t,train_score

