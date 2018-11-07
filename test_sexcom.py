import requests
import argparse
import time
import os

PyTorch_REST_API_URL = 'http://0.0.0.0:5000/predict'
def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()
    print(r)
    return r['predictions']

start = time.clock()
pred=predict_result('./2.jpg')
elapsed = (time.clock() - start)
print(elapsed)
print(pred)
import requests
import argparse
import time
import os
import pandas as pd
import numpy as np
from PIL import Image

print("Time used:",elapsed)
print('test access promised..')
print('starting test on training image')
TN=0
TP=0
FP=0
FN=0

a=[1,2,3,4,5,6,7,8,9]
testnp=np.array(a)
savepath='sexcomeresult_dense.csv'
rootdir = '/home/claude.duan/data/sexcom'
list = os.listdir(rootdir)
for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path):
            try:
                start = time.clock()
                pred=predict_result(path)
                elapsed = (time.clock() - start)
                if pred[0]['Common']>pred[0]['Porn']:
                    label=0
                else:
                    label=1
                if label == 0:
                    FN = FN+1
                else:
                    TP = TP+1
                ed = [label, pred[0]['Common'] , pred[0]['Porn'], elapsed, path, TN, TP, FP, FN]
                testnp=np.vstack((testnp,np.array(ed)))
                test_data=pd.DataFrame(testnp,columns=['label','common','porn','time','path','TN','TP','FP','FN'])
                test_data.to_csv(savepath) 
            except:
                print('The image is just a link')