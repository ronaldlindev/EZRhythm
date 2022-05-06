import paramiko
from datetime import datetime
from datetime import date
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
    data = pd.read_csv(fr'C:\Users\YouConfusedYet\Desktop\ECG Webserver\webserver\webs\files\2022-04-02.txt', header = None).values.tolist()
    data = data[0]
    
    cleaned = [ x for x in data if isinstance(x, int)]
    print(len(cleaned))
    # for i in range(12):
    #     cleaned += cleaned
    #     print(len(cleaned))
    
    print(f'file cleaned at {datetime.now()}')
    DenoisedECG = highpass(notch(cleaned, sampFreq = 128), cutoff = 1, fs = 128) # baseline wanderingremoval and powerline interfence removal
    
    plt.plot(DenoisedECG)
    plt.show()
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
    plt.plot(sinusMedian.T)
    plt.plot(x.iloc[0,:])
    plt.show()
    return sinusMedian

import matlab.engine
def predictSinusnoSinus(df):
    df = df.values.tolist()
    eng = matlab.engine.start_matlab()
    param = eng.importONNXFunction(r"C:\Users\YouConfusedYet\Desktop\ECG Webserver\webserver\webs\models\sinusnosinus.onnx", "cum")
    for ele in df:
        A = matlab.double(ele, size = [300,1])
        SOFTMAX1000 = eng.cum(A,param)
        label = conversion[np.argmax(SOFTMAX1000[0][0])]
        print(label)
        ele.insert(0,label)
    eng.close()
    return df
    
def finalClassification(df):
    eng = matlab.engine.start_matlab()
    param = eng.importONNXFunction(r"C:\Users\YouConfusedYet\Desktop\ECG Webserver\webserver\webs\models\modelwdiff", "piss")
    A = matlab.double(df.values.tolist(), size = [300,1])
    SOFTMAX1000 = eng.piss(A,param)
    print(SOFTMAX1000)
    print(conversion[np.argmax(SOFTMAX1000[0][0])])
    eng.close()
    

# getfile()
df = fileToSegmentTodf()
# print(df.head())
# # df.to_hdf("bruh")
# print(plt.plot(df.values.tolist()[0])) 
# plt.show()  
conversion = {0:'N', 1:'S'}
l = predictSinusnoSinus(df)
print(l)
# N = []
# S = []
# for ele in l:
#     if ele[0] == 'S':
#         S.append(ele)
#     else: 
#         N.append(ele)
# for ele in S:
#     plt.plot(ele)
#     plt.title('S')
#     plt.show()
#     plt.clf()

# for ele in N:
#     plt.plot(ele)
#     plt.title('N')
#     plt.show()
#     plt.clf()




end_time = datetime.now() 
print('Duration: {}'.format(end_time - start_time))


# median = sinusExtractorandMedian(df)
