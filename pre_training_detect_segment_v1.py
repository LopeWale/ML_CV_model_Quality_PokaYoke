"""
Create a custom convolutional neural network system that can uses an object detection model to train on specify images which is then use the weights from the first model in another model to detect all the instances of object trained in the first model in other different images with a object segmentation model with high MAP accuracy and low computational cost and fast computational speed
"""

import torch 
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.path as mplPath
from torchvision import models

#----------------------------------
#--------------Parsing-data-into-train-and-evaluation===================

KERNEL_SIZE = 3

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model_1 = models.vgg19(pretrained=True) # pretrained AlexNet model --> this contains all the model architecture of VGG19, we do not need to build the model architecture on our own
        #print(self.model_1)
        self.features = self.model_1.features
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        #modify the last classifier layer from 1000 to 2
        #Randomly initialize the last fully connected layer
        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
        self.classifier[6] = nn.Linear(4096, 2)

        self.model_1.classifier = self.classifier
        self.model_2 = models.vgg19(pretrained=True)

        self.con_features = nn.Sequential(*list(self.model_2.features.children())[:-1])
        self.con_features[31] = nn.Sequential(nn.Conv2d(512, 214, kernel_size=3, stride=1, padding=1),nn.ConvTranspose2d(214, 214, 2, stride=2, bias=False))
        self.con_features[31].weight = nn.Parameter(self.init(214,214))

    def init(self, out_ch,in_ch):
        w = np.random.randn(out_ch,in_ch, 3, 3)
        return torch.from_numpy(w).float()
        
    def forward(self, x, p = -1):
        if isinstance(x,np.ndarray):
            x = torch.from_numpy(x).float().cuda()
        x = F.interpolate(x, size=256)
        out = Variable(x).cuda()
        out1 = self.model_1(out)
        if p == -1:
            out2 = self.model_2(out)
        else:
            out2 = self.con_features(out)
            num_bins = 214
            batch_size = out2.size(0)
            semseg = torch.zeros(batch_size,num_bins,224, 224).cuda()
            ind_cnt = torch.ones(num_bins,1,1).cuda()
            for j in range(len(out2[0][0][0])):
                for k in range(len(out2[0][0][0][0])):
                    ind = p[0][j][k]
                    semseg[:,ind,j:j+1,k:k+1] += out2[:,:,j,k].unsqueeze(1)
                    ind_cnt[ind] += 1
            semseg = semseg.sum(1, keepdim=True) / ind_cnt
            semseg = out2
        return F.softmax(out1, dim = 1), out2

model = MyModel()
model = model.cuda()

optimizer = torch.optim.SGD(model.model_1.parameters(), lr=0.000001) #optimizer used is Adam optimizer
optimizer2 = torch.optim.SGD(model.model_2.parameters(), lr=0.000001) #optimizer used is Adam optimizer

loss_fn = torch.nn.CrossEntropyLoss().cuda() #Loss function is Cross entropy loss
loss_fn_2 = torch.nn.MSELoss().cuda() #Loss function is Cross entropy loss

input_size = 224
means     = [0.485, 0.456, 0.406]
stds      = [0.229, 0.224, 0.225]

def resize_and_crop(image, size=input_size):
    w, h = image.size
    if w > h:
        osize = (size * w // h, size)
    else:
        osize = (size, size * h // w)
    image = image.resize(osize, 0)
    new_w, new_h = image.size
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    right = (new_w + size) // 2
    bottom = (new_h + size) // 2
    image = image.crop((left, top, right, bottom))
    return image

def resize_and_centercrop(image, size=input_size):
    w, h = image.size
    if w > h:
        osize = (size * w // h, size)
    else:
        osize = (size, size * h // w)
    image = image.resize(osize, 0)
    new_w, new_h = image.size
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    right = (new_w + size) // 2
    bottom = (new_h + size) // 2
    image = image.crop((left, top, right, bottom))
    return image


#----------------------------------
#--------------------------Switching-of-data-in-Training-Phase--------------------  
def one_hot(pixels):
    class_width = 8
    npixels = len(pixels)
    x = np.zeros([npixels, 2 + class_width * class_width])
    for i, p in enumerate(pixels):
        x[i,0] = min(p[0] / input_size, 1)
        x[i,1] = min(p[1] / input_size, 1)

        #bin_size = input_size // class_width
        #xpixel = min(p[0], input_size - 1)
        #ypixel = min(p[1], input_size - 1)
        #x[i,0] = 0.5
        #x[i,1] = 0.5
        #x[i,2 + xpixel // bin_size + (ypixel // bin_size) * class_width] = 1
    return x.transpose((0,3,1,2)) 


def collate_fn(batch):
    """
    where you pre-process the data
    """
    batch_size = len(batch)
    images1 = []
    images2 = []
    keypoints = []
    for b in batch:
        images1.append(np.array(resize_and_centercrop(b[0]).getdata(), dtype=np.float32) / 255.0)
        images2.append(np.array(resize_and_centercrop(b[2]).getdata(), dtype=np.float32) / 255.0)
        keypoints.append(one_hot(b[1]))
    X1 = np.array(images1).transpose((0, 3, 1, 2))
    X2 = np.array(images2).transpose((0, 3, 1, 2))
    X1 = (X1 - means) / stds
    X2 = (X2 - means) / stds
    return torch.from_numpy(X1).float().cuda(), torch.from_numpy(X2).float().cuda(), torch.from_numpy(np.array(keypoints)).float().cuda()

#=======================================================================
#------------------------Switching-of-data-in-evaluation-phase-------------------

def convert_to_coor(output, batch_size = 4):
    num_bins = 214
    semseg = output
    bin_size = input_size // num_bins
    ks = bin_size // 2
    coors = []
    for i in range(batch_size):
        coors_i = []
        for j in range(len(semseg[0])):
            #print("j is ", j)
            for k in range(len(semseg[0][0])):
                #print("k is ", k)
                #print("channels are ",  np.argmax(semseg[i,j,k].cpu().detach().numpy()))
                if np.argmax(semseg[i,j,k].cpu().detach().numpy())!=0 and j!=0 and j!=len(semseg[0])-1 and k!=0 and k!=len(semseg[0][0])-1:
                    for m in range(-2,2):
                        for n in range(-2,2):  
                            if np.argmax(semseg[i,j+m,k+n].cpu().detach().numpy())==np.argmax(semseg[i,j,k].cpu().detach().numpy()) and j+m!=len(semseg[0])-1 and k+n!=len(semseg[0][0])-1:
                                coor = (j*bin_size+ks, k*bin_size+ks)
                                coors_i.append(coor)
        coors.append(coors_i)
        
    coors = np.array(coors)
    return coors
    
def iou(pixels1, pixels2):
    poly1 = mplPath.Path(pixels1)
    poly2 = mplPath.Path(pixels2)
    try:
        intersection = poly1.intersection(poly2)
        intersection = len(intersection.vertices)
    except Exception as e:
        intersection = 0
    union = len(pixels1) + len(pixels2) - intersection
    return intersection / union

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate(model, cache_hit={}, batch_size=4, threshold=0.5):
    image_names = os.listdir('lfw_funneled')
    n_per_image = 4
    step_size = 4
    n_images = len(image_names)
    n_samples = n_per_image * n_images
    n_batches = -(-n_samples // batch_size)
    all_ious = []
    all_recalls = []
    all_precisions = []
    all_aps = []
    batch = []
    for i in range(n_batches):
        if not batch and i != 0:
            break
        batch = []
        for b in range(batch_size):
            image_ind = (i * batch_size + b) // n_per_image
            sample_ind = (i * batch_size + b) % n_per_image
            person = ' '.join(image_names[image_ind].split('_')[:-1])
            person_dir = os.path.join('lfw_funneled', person)
            image_name = os.path.join(person_dir, image_names[image_ind])
            df_name = os.path.join('lfw_funneled', person + '.txt')
            df = pd.read_csv(df_name, delim_whitespace=True, header=None)
            df = df.sort_values([0])
            df = df.values
            image1 = cv2.imread(image_name)
            image2 = cv2.imread(image_name)
            image1 = cv2.resize(image1, (0,0), fx=0.5, fy=0.5)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2 = cv2.resize(image2, (0,0), fx=0.5, fy=0.5)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            
            if image_name in cache_hit:
                pixels = cache_hit[image_name]
            else:
                pixels = df[sample_ind*step_size:(sample_ind+1)*step_size:2,1:].astype(np.uint32)
                pixels = pixels[:, [1,0]]
                pixels[:,1] = image1.shape[1] - pixels[:,1]
                cache_hit[image_name] = pixels
            batch.append((image1, pixels, image2))
        x, y, _ = collate_fn(batch)
        output1, output2 = model(x)
        predictions1 = output1.max(1)[1]
        predictions2 = output2.max(1)[1]
        predictions2 = predictions2.cpu().detach().numpy()
        predictions1 = predictions1.cpu().detach().numpy()
        coors = convert_to_coor(output2)
        coors = coors.astype(int)
        """
        df = pd.read_csv('lfw_funneled/George_W_Bush/George_W_Bush_0001.txt', delim_whitespace=True, header=None)
        df = df.sort_values([0])
        df = df.values
        """
        for pred, gt, ious, (image1, pixels, image2) in zip(predictions2, y, coors, batch):
            pred = pred[np.where(pred != 23)]
            gt = gt[np.where(gt != 23)]
            ious_values = []
            for i in range(gt.shape[0]):
                try:
                    ious_values.append(ious[pred==gt[i]].max())
                except:
                    ious_values.append(0)
            if ious_values == []:
                ious_values = [0]
            ious_values = np.array(ious_values)
            all_ious.append(ious_values)
            tp = ious_values >= threshold
            tp = tp.sum()
            fp = tp - ious_values.sum()
            all_recalls.append(tp / gt.shape[0])
            all_precisions.append(tp / (tp + fp).clip(0,1))
            for coordinate in coors:
                #image1 = cv2.circle(image1, (coordinate[0], coordinate[1]), radius=1,
                #                    color=(0, 0, 255), thickness=2)
                cv2.line(image1, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), color=(0,0,255), thickness=2)
            for i in pixels:
                cv2.line(image2, (i[0], i[1]), (i[2], i[3]), color=(0,0,255), thickness=2)
            image1 = Image.fromarray(image1)
            image2 = Image.fromarray(image2)
            image1.show()
            image2.show()
            cv2.waitKey(5)
            """
            #image2 = cv2.circle(image2, (int(i[0]), int(i[1])), radius=1, 
                                #color=(0, 0, 255), thickness=2)
            image2 = cv2.rectangle(image2, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), color=(0,255,0), thickness=2)
            """
    all_ious = np.concatenate(all_ious, 0)
    all_ious = np.sort(all_ious)
    for i in range(40, 100, 5):
        n = np.where(all_ious > i/100)[0].shape[0]
        acc = np.mean(np.array(all_ious > i/100))
        print(n, i/100, f'{acc * 100:.0f}%')
    precisions = np.array(all_precisions)
    recalls = np.array(all_recalls)
    print(all_ious)
    print(precisions)
    print(recalls)
    ap = average_precision_score(recalls, precisions)
    print('AP: ', ap)
    #print(all_ious.shape, precisions.shape, recalls.shape)
    """
    plt.figure()
    plt.plot(recalls, precisions)
    plt.grid(1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    """
model = torch.load('../latest.pt')

evaluate(model)