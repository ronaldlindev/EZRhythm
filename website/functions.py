import paramiko
from datetime import datetime
from datetime import date
import numpy as np
start_time = datetime.now()


def getfile():
    ip = '192.168.50.222'
    port = 22
    username = 'pi'
    password = 'pi'
    transport = paramiko.Transport((ip, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    import sys
    path = fr'C:\Users\YouConfusedYet\Desktop\ECG Webserver\webserver\webs\files\{date.today()}.txt'
    sftp.get(remotepath=f'/home/pi/Documents/{date.today()}.txt', localpath=path)
    sftp.close()
    transport.close()
    print(f'file retrieved at {datetime.now()}')
    return

# getfile()

from scipy import signal
from scipy.signal import find_peaks
from datetime import date
import math
from math import floor, ceil
import pandas as pd
import pywt
import statistics
import numpy as np
import matplotlib.pyplot as plt
# import onnx
# from onnx_tf.backend import prepare
import os
# import onnx
# from onnx_tf.backend import prepare



PATH = r'C:\Users\YouConfusedYet\Desktop\ECG Webserver\webserver\webs\files\2022-03-24.txt'
def segmenter(data, beat, RRRandMidLabels = pd.DataFrame(index = None), PAD = 300): # data is SIG_II file or same format, beat is BEAT file or same format 
    beat = beat[0].tolist()
    for ele in beat:
        if beat.index(ele) + 2 < len(beat) - 1:    
            elePlusOne = beat[beat.index(ele) + 1] # middle element
            elePlusTwo = beat[beat.index(ele)  + 2]
            RRRsegment = data[int(ele):int(elePlusTwo)].tolist() #segment from first R peak to third
            if len(RRRsegment) < PAD:
                rpad = PAD/2 - (int(elePlusTwo) - int(elePlusOne))
                lpad = PAD/2 - (int(elePlusOne) - int(ele))
                RRRsegment = np.pad(RRRsegment, constant_values = (0,0), pad_width = (int(lpad), int(rpad))).tolist()

                RRRandMidLabels = pd.concat([RRRandMidLabels, pd.DataFrame(RRRsegment, index = None)], ignore_index = True, axis = 1)
                # RRRandMidLabels.append([elePlusOne, RRRsegment])
    return RRRandMidLabels # note that have not been trimmed and thus are just RRR segments.
            
def notch(Signal, sampFreq = 128, notchFreq = 60.0, qualityFactor = 4.0): # no fucking clue what this q factor means but random number dw
    b, a = signal.iirnotch(notchFreq, qualityFactor, sampFreq)
    return signal.filtfilt(b, a, Signal)
def highpass(Signal, cutoff, fs, order = 5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, Signal)

def fileToSegmentTodf():
    # data = pd.read_csv(fr'C:\Users\YouConfusedYet\Desktop\ECG Webserver\webserver\webs\files\2022-04-02.txt', header = None).values.tolist()
    # data = data[0]
    # cleaned = [x for x in data if isinstance(x, int)]
    cleaned = np.load("files/100_SIG_II.npy")
    
    print(f'file cleaned at {datetime.now()}')
    DenoisedECG = highpass(notch(cleaned, sampFreq = 128), cutoff = 1, fs = 128) # baseline wanderingremoval and powerline interfence removal
    peak_h = find_peaks(DenoisedECG, height = 0.25, distance = 6 )
    # print(peak_h[0])
    (cA, cD) = pywt.dwt(DenoisedECG, 'sym4', mode = 'zero') # returns approximation (cA) and detail (cD) coefficients.
    print(f'wavelet calculated at {datetime.now()}')
    #threshold calculation from https://www.hindawi.com/journals/jhe/2017/4901017/ 2.1
    t = math.sqrt((2 * math.log(len(cD)))/2) * statistics.median(abs(cD)/0.6745) #universal threshold selection method do you include cA??
    
    cD_hat_hard = pywt.threshold(cD, t, 'hard')
    DenoisedECG = pywt.idwt(cA, cD_hat_hard, 'sym4', mode = 'zero')
    DenoisedECG = (DenoisedECG - DenoisedECG.min(axis=0)) / (DenoisedECG.max(axis=0) - DenoisedECG.min(axis=0)) # zero normalization?

    df = segmenter(DenoisedECG, peak_h).T.apply(pd.to_numeric, errors = 'ignore')
    print(f'segmented {datetime.now()}')
    return df


def sinusExtractorandMedian(RRRandMidLabels):
    x = RRRandMidLabels
    sinus = x[x.iloc[:,0] == 'S']
    sinusMedian = sinus.iloc[:,1:].median()
    sinusMedian = (sinusMedian - sinusMedian.min(axis=0)) / (sinusMedian.max(axis=0) - sinusMedian.min(axis=0)) # zero normalization?
    return sinusMedian

import matlab.engine
def predictSinusnoSinus(df):
    conversion = {0:'N', 1:'S'}
    df = df.values.tolist()
    eng = matlab.engine.start_matlab()
    param = eng.importONNXFunction(r"models\sinusnosinus.onnx", "cum")
    for ele in df:  
        A = matlab.double(ele, size = [300,1])
        SOFTMAX1000 = eng.cum(A,param)
        label = conversion[np.argmax(SOFTMAX1000[0][0])]
        # print(label)
        ele.insert(0,label)
    eng.close()

    return df
    
def finalClassification(df):
    df = df.values.tolist()
    conversion = {0:'A', 1: 'B', 2: 'N', 3: 'Q', 4: 'S', 5: 'V'}
    eng = matlab.engine.start_matlab()
    param = eng.importONNXFunction(r"models\finalwdiff.onnx", "piss")
    for ele in df:
        se = matlab.double([ele[300]])
        data = matlab.double([ele[0:299]])
        SOFTMAX1000 = eng.piss(se, data,param)
        print(SOFTMAX1000)
        label = conversion[np.argmax(SOFTMAX1000[0][0])]
        print(label)
        ele.insert(0, label)
    eng.close()
    return df


from sklearn.metrics import mean_squared_error


# # getfile()
def full():
    df = fileToSegmentTodf()
    df = predictSinusnoSinus(df) # returns list
    bruh = pd.DataFrame(df)
    med = sinusExtractorandMedian(bruh)
    bruht = bruh.apply(pd.to_numeric, errors = 'ignore')
    drop_prev_labels = bruht.drop(bruht.columns[0], axis = 1)
    l = []
    sample = [0.0 for i in range(300)]
    for index, row in drop_prev_labels.iterrows():
                if med.isnull().values.any() == True:
                    l.append(mean_squared_error(row.values.tolist()[100:200], sample[100:200]))
                else:
                    l.append(mean_squared_error(row.values.tolist()[100:200], med[100:200]))
    drop_prev_labels.insert(300,301,l)
    print(drop_prev_labels.head())
    df = finalClassification(drop_prev_labels)
    print(df.head)
full()




end_time = datetime.now() 
print('Duration: {}'.format(end_time - start_time))


