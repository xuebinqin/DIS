import os
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models import *


if __name__ == "__main__":
    dataset_path="../demo_datasets/your_dataset"  #Your dataset path
    model_path="../saved_models/IS-Net/isnet-general-use.pth"  # the model path
    result_path="../demo_datasets/your_dataset_result"  #The folder path that you want to save the results
    input_size=[1024,1024]
    net=ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
    net.eval()
    im_list = glob(dataset_path+"/*.jpg")+glob(dataset_path+"/*.JPG")+glob(dataset_path+"/*.jpeg")+glob(dataset_path+"/*.JPEG")+glob(dataset_path+"/*.png")+glob(dataset_path+"/*.PNG")+glob(dataset_path+"/*.bmp")+glob(dataset_path+"/*.BMP")+glob(dataset_path+"/*.tiff")+glob(dataset_path+"/*.TIFF")
    for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
        print("im_path: ", im_path)
        im = io.imread(im_path)
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_shp=im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
        im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
        image = torch.divide(im_tensor,255.0)
        image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

        if torch.cuda.is_available():
            image=image.cuda()
        result=net(image)
        result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        im_name=im_path.split('/')[-1].split('.')[0]
        io.imsave(os.path.join(result_path,im_name+".png"),(result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8))
