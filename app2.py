import flask
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from models import pretrained
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import io

app = flask.Flask(__name__)

def load_model(trainpath):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #best_acc = 0  # best test accuracy
    #start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    net = pretrained.densenet_Porn_Com()
    net = net.to(device)
    #net = torch.nn.DataParallel(net)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    else:
        net = torch.nn.DataParallel(net)
    assert os.path.isfile(trainpath), 'Error: no trained model directory found!'
    if device == 'cpu':
        checkpoint = torch.load(trainpath,map_location='cpu')
        #checkpoint = torch.load(trainpath) 
    else:
        checkpoint = torch.load(trainpath)     
    net.load_state_dict(checkpoint['net'])
    return net

def image_to_tensor(pil_image):
    #resize image
    
    # transform it into a torch tensor
    loader = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return loader(pil_image).unsqueeze(0) #need to add one dimension, need to be 4D to pass into the network 


def get_porn_probability(all_probabilities):
    probs=all_probabilities.data.numpy()[0]
    probs.tolist()
    #print(probs)
    return probs
    
def dense_model():
    global model 
    model = load_model('./train/dense/dense.plk')
    model.eval()
    

@app.route("/predict", methods=["POST"])

def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {'success':False}
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image)).convert("RGB")

            # Preprocess the image and prepare it for classification.
            image = image_to_tensor(image)
            image = Variable(image)
            
            data['predictions'] = list()
            #model = load_model('/Users/duanyiqun/Documents/Cap_porndetection/Sparse_Dense_Module/train/sdmnv5/sdmnv5.plk')
            #model.eval()
            probabilities = model(image)
            prob = F.softmax(probabilities,dim=1)
            num_prob=get_porn_probability(prob.cpu())
            
            r = {"Common":float(num_prob[0]) , "Porn": float(num_prob[1])}
            data['predictions'].append(r)
            # Classify the input image and then initialize the list of predictions to return to the client.
            data['success'] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)

if __name__ == '__main__':
    dense_model()
    app.run(debug=True, host='0.0.0.0')