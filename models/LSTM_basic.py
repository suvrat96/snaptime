from snaptime import LSTM_monolithic
import numpy as np
from snaptime_helper import FillData
import pickle
# from time import time
import time
import os

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.regularizers import l1
import tensorflow as tf
from keras.optimizers import Adam

#from keras.backend.tensorflow_backend import set_session
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#set_session(tf.Session(config=config))


model = Sequential()
model.add(LSTM(256,input_shape=(100,1397)))#dropout_W=0.95,dropout_U=0.95))
model.add(Dense(2))
model.add(Activation('sigmoid'))
opt = Adam(lr=0.0015)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



fpr,tpr,thres,test_score,fpr_t,tpr_t,thres_t,train_score = LSTM_monolithic.run_LSTM('/dfs/scratch0/david/vw/driver_date_V2',\
        np.array([i for i in xrange(1397)]),\
        np.array([1173]),\
        100,\
        10,\
        3600000,\
        100,\
        lambda y : y[0] == 1,\
        5000,\
        model,\
        10, testAll=False)

print test_score,train_score


output = (fpr,tpr,thres,test_score,fpr_t,tpr_t,thres_t,train_score)
pickle.dump(output, open('outputs.pkl', 'wb'))
