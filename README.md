# Deploy

# Requirement
```Pytorch 0.4, flask, PIL```   
To use this model, you should establish a train folder and under that to place your plk files. We didn't provide our model here.

```sh
sudo nvidia-docker run -it -p 5000:5000 yourname/pytorch:yourtag 
cd Deploy
python app2.py
```
   
```python
client example:

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
    return r

start = time.time()
pred=predict_result('./2.jpg')
elapsed = (time.time() - start)
print(elapsed)
#print(pred)
```
