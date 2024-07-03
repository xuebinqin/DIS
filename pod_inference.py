import os
import sys
import time
script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(script_dir)

print(os.getcwd())
import cv2
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_loader_cache import normalize, im_reader, im_preprocess 
from models import *
from maskHelper import maskImage
import uuid 
#Helpers
hypar = {} 

global net 

net = None  
masker = maskImage()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image


transform =  transforms.Compose([GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0])])

def load_image(image_path, hypar):
    # print(hypar)
    im = cv2.imread(image_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im,255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0) # make a batch of image, shape


def build_model(hypar,device):
    global net 
    net = hypar["model"]#GOSNETINC(3,1)

    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if(hypar["restore_model"]!=""):
        net.load_state_dict(torch.load(hypar["model_path"]+"/"+hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()  
    return net

    
def predict(  inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    global net 
    net.eval()

    if(hypar["model_digit"]=="full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

  
    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device) # wrap inputs in Variable
   
    ds_val = net(inputs_val_v)[0] # list of 6 results

    pred_val = ds_val[0][0,:,:,:] # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[0][0],shapes_val[0][1]),mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi) # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8) # it is the mask we need


def load_hyperparameters(model_dir, weights_path):
    global net 
    hypar["interm_sup"] = False 
    hypar["model_digit"] = "full" 
    hypar["seed"] = 0
    hypar["cache_size"] = [640, 640] ## cached input spatial resolution, can be configured into different size
    hypar["input_size"] = [640, 640] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
    hypar["crop_size"] = [640, 640] ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation


    hypar["model_path"] = model_dir ## load trained weights from this path dis-background-removal/saved_models/gpu_itr_250000_traLoss_0.0515_traTarLoss_0.0021_valLoss_0.0519_valTarLoss_0.0021_maxF1_1.0_mae_0.0009_time_0.026797.pth
    hypar["restore_model"] = weights_path 
    hypar["model"] = ISNetDIS()

    net = build_model(hypar, device)


def inference(image, threshold_value = 240, max_value = 255):

    image_tensor, orig_size = load_image(image, hypar) 
    mask = predict(image_tensor, orig_size, hypar, device)
    # gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)
    return binary_image 


def predict_xyxy(image, threshold_value = 240, max_value = 255):
    image_tensor, orig_size = load_image(image, hypar) 
    mask = predict(image_tensor, orig_size, hypar, device)

    _, binary_image = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)

    points_list = masker.maskProcessor(binary_image)
    bounding_boxes = masker.get_xyxy(points_list)
    bounding_boxes = bounding_boxes[:1]
    return bounding_boxes, binary_image

if __name__ == "__main__":
    load_hyperparameters("saved_models/", "best_model.pth")
    import cv2 
    image_path = "5333eee2-37ac-11ef-88f5-87f7348bea41.jpg"

    bounding_boxes, binary_image = predict_xyxy(image_path)
    print(bounding_boxes)