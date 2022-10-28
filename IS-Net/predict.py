import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from data_loader_cache import normalize, im_reader, im_preprocess
from models.isnet import ISNetDIS

from cog import BasePredictor, Path, Input

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Predictor(BasePredictor):
    def setup(self):
        self.net = ISNetDIS()
        self.net.load_state_dict(torch.load("isnet.pth", map_location=device))
        self.net.to(device)
        self.net.eval()

    def predict(
            self,
            input_image: Path = Input(description="Image to segment."),

    ) -> Path:
        cache_size = [1024,1024]
        image, orig_size = load_image(str(input_image), cache_size)

        image = image.type(torch.FloatTensor)

        image = Variable(image, requires_grad=False).to(device)  # wrap inputs in Variable

        ds_val = self.net(image)[0]  # list of 6 results

        pred_val = ds_val[0][0, :, :, :]  # B x 1 x H x W    # we want the first one which is the most accurate prediction

        ## recover the prediction spatial size to the orignal image size
        pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val, 0), (orig_size[0][0], orig_size[0][1]), mode='bilinear'))

        ma = torch.max(pred_val)
        mi = torch.min(pred_val)
        pred_val = (pred_val - mi) / (ma - mi)  # max = 1

        if device == 'cuda':
            torch.cuda.empty_cache()

        output_path = "output.png"
        save_image(pred_val, output_path, normalize=True)

        return Path(output_path)


class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


def load_image(im_path, cache_size):
    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, cache_size)
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])
    return transform(im).unsqueeze(0), shape.unsqueeze(0)  # make a batch of image, shape