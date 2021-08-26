import os
import sys
import base64
import json
import cv2
import boto3
import numpy as np
from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.models.modnet import MODNet
input_path = 'test'
output_path = 'output'
ckpt_path='./modnet_photographic_portrait_matting.ckpt'

def lambda_handler(event, context):
    # define cmd arguments
    outputp=output_path
    ckptp=ckpt_path
    try:
      url=event['image']
    #   response = requests.get(event['image'])
    except:
      pass
    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    modnet.load_state_dict(torch.load(ckptp, map_location=torch.device('cpu')))
    modnet.eval()

    # inference images

    imi = Image.open(requests.get(url, stream=True).raw)
    basewidth=1500
    baseheight=1500
    if imi.size[0]>1000 and imi.size[1]>1500:
        wpercent = (basewidth/float(imi.size[0]))
        hsize = int((float(imi.size[1])*float(wpercent)))
        imi = imi.resize((basewidth,hsize), image.ANTIALIAS)
        

    # unify image channels to 3
    imi = np.asarray(imi)
    if len(imi.shape) == 2:
        imi = imi[:, :, None]
    if imi.shape[2] == 1:
        imi = np.repeat(imi, 3, axis=2)
    elif imi.shape[2] == 4:
        imi = imi[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(imi)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    # print(im_b, im_c, im_h, im_w)
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im, True)
    
    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    im_name='1'
    
#     matte_name = im_name.split('.')[0] + '.png'
#     # print(matte.shape)
#     Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(outputp, matte_name))

#     matte = Image.open(os.path.join(outputp, matte_name))
#     # print(matte.shape)
#     matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
#     w, h = im_w, im_h
#     rw, rh = 800, int(h * 800 / (3 * w))
#     foreground = imi * matte + np.full(imi.shape, 255) * (1 - matte)
#     cv2.imwrite(os.path.join(outputp, matte_name),foreground)
#     Image.fromarray((foreground.astype('uint8'))).save(os.path.join(outputp, '2.png'))
#     try:
#       s3 = boto3.client("s3")
#       s3.put_object(Bucket='abhishekimagebucket', Key="12", Body=open("/tmp/test2.mp4", "rb").read())
#     except:
#       pass
    outputp='/tmp/'
    matte=matte*255
    matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
    Image.fromarray((foreground.astype('uint8'))).save(os.path.join(outputp, '2.png'))
    try:
        s3 = boto3.client('s3',
                          aws_access_key_id: 'AKIA3TO5KM7D3HRVTRRV',
                          aws_secret_access_key: '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',
                          region_name: 'us-east-1')
        s3.put_object(Bucket='abhishekimagebucket', Key="123.png", Body=open(os.path.join(outputp,'2.png'), "rb").read())
    except:
        pass
    
    
    return 'https://abhishekimagebucket.s3.amazonaws.com/123.png'

    
# print(lambda_handler("https://images.unsplash.com/photo-1534528741775-53994a69daeb?ixid=MnwxMjA3fDB8MHxzZWFyY2h8OXx8cG9ydHJhaXR8ZW58MHx8MHx8&ixlib=rb-1.2.1&w=1000&q=80",2))
