# -*- coding: utf-8 -*-

import torch
torch.cuda.is_available()

import pandas as pd
import numpy as np
from array import *
#from PIL import Image
import cv2
import os
import random

"""Sorry....change the path to 64 * 64 dimension

"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
print("Time to load train arrays")
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/X_train_1.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/y_train_1.npy"
X_train = np.load(path_x)
y_train = np.load(path_y)

print(np.unique(y_train,return_counts=True))

# Commented out IPython magic to ensure Python compatibility.
# %%time
print("Time to load val arrays")
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/X_val_1.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/y_val_1.npy"
X_val = np.load(path_x)
y_val = np.load(path_y)

print(np.unique(y_val,return_counts=True))

# Commented out IPython magic to ensure Python compatibility.
# %%time
print("Time to load test arrays")
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/X_test_1.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/y_test_1.npy"
X_test = np.load(path_x)
y_test = np.load(path_y)

print(np.unique(y_test,return_counts=True))

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def extract_sample(n_way, n_support, n_query, datax, datay):
  sample = []
  label = []
  K = np.random.choice(np.unique(datay), n_way, replace=False)
  for cls in K:
    #datax_cls represents the images data 64*64*3 corresponding to a class
    datax_cls = datax[datay == cls]
    #random permutations on datax_cls
    perm = np.random.permutation(datax_cls)
    #selects n_support + n_query
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
    for i in range(n_support+n_query):
        label.append(cls)
  sample = np.array(sample)
  sample = torch.from_numpy(sample).float()
  #sample size is: torch.Size([3, 6, 64, 64, 3])
  sample = sample.permute(0,1,4,2,3)
  #after permute: torch.Size([3, 6, 3, 64, 64])
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'n_label': label
      })

def extract_sample_test(K, n_way, n_support, n_query, datax, datay):

  sample = []
  label = []
  #K = np.choice(np.unique(datay), n_way, replace=False)
  #K = df_mod_test['Finding Labels'].unique()
  for cls in K:
    #datax_cls represents the images data 64*64*3 corresponding to a class
    datax_cls = datax[datay == cls]
#     print("inside test:", cls)
    #random permutations on datax_cls
    perm = np.random.permutation(datax_cls)
    #selects n_support + n_query
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
#     print("inside test 1:", cls)
    for i in range(n_support+n_query):
#         print("inside test 2:", cls)
        label.append(cls)
  sample = np.array(sample)
  sample = torch.from_numpy(sample).float()
  #sample size is: torch.Size([3, 6, 64, 64, 3])
  sample = sample.permute(0,1,4,2,3)
  #after permute: torch.Size([3, 6, 3, 64, 64])
#   print("inside test:", label)
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'n_label': label
      })

def display_sample(sample):
  """
  Displays sample in a grid
  Args:
      sample (torch.Tensor): sample of images to display
  """
  #need 4D tensor to create grid, currently 5D
  sample_4D = sample.view(sample.shape[0]*sample.shape[1],*sample.shape[2:])
  #make a grid
  out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
  plt.figure(figsize = (16,7))
  plt.imshow(out.permute(1, 2, 0))
  plt.savefig("split1_img")

sample_example = extract_sample(3, 4, 1, X_test, y_test)
print(sample_example['n_label'])
display_sample(sample_example['images'])

"""## **Required Methods**"""

def one_hot(indices, depth, dim=-1, cumulative=True):
    new_size = []
    for ii in range(len(indices.size())):
        if ii == dim:
            new_size.append(depth)
        new_size.append(indices.size()[ii])
    if dim == -1:
        new_size.append(depth)

    out = torch.zeros(new_size)
    indices = torch.unsqueeze(indices, dim)
    out = out.scatter_(dim, indices.data.type(torch.LongTensor), 1.0)
    return Variable(out)

def compute_protos(h, probs):
    h = torch.unsqueeze(h, 2)       # [B, N, 1, D]
    probs = torch.unsqueeze(probs, 3)       # [B, N, nClusters, 1]
    prob_sum = torch.sum(probs, 1)  # [B, nClusters, 1]
    zero_indices = (prob_sum.view(-1) == 0).nonzero()

    if torch.numel(zero_indices) != 0:
        values = torch.masked_select(
            torch.ones_like(prob_sum), torch.eq(prob_sum, 0.0))
        prob_sum = prob_sum.put_(zero_indices, values)
    protos = h*probs    # [B, N, nClusters, D]
    protos = torch.sum(protos, 1)/prob_sum
    return protos

def estimate_lamda(proto_mean):
  alpha = 0.1
  dim = 100
  init_sigma_l = 5
  log_sigma_l = torch.log(torch.FloatTensor([init_sigma_l]))
  log_sigma_l = Variable(log_sigma_l, requires_grad=True).cuda()
  sigma = torch.exp(log_sigma_l).detach().cpu().numpy().data[0]
  rho = proto_mean[0].var(dim=0)
  rho = rho.mean().detach().cpu().numpy()
  lamda = -2*sigma*np.log(alpha) + dim*sigma*np.log(1+rho/sigma)
  return lamda

def compute_distances(protos, example):
        dist = torch.sum((example - protos)**2, dim=2)
        return dist

def add_cluster(nClusters, protos, radii, ex=None):
        nClusters += 1
        bsize = protos.size()[0]
        dimension = protos.size()[2]
        init_sigma_l = 5
        log_sigma_l = torch.log(torch.FloatTensor([init_sigma_l]))
        log_sigma_l = Variable(log_sigma_l, requires_grad=True).cuda()
        zero_count = Variable(torch.zeros(bsize, 1)).cuda()
        d_radii = Variable(torch.ones(bsize, 1), requires_grad=False).cuda()
        d_radii = d_radii * torch.exp(log_sigma_l)

        new_proto = ex.unsqueeze(0).unsqueeze(0).cuda()

        protos = torch.cat([protos, new_proto], dim=1)
        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, radii

def delete_empty_clusters(tensor_proto, prob, radii, targets, eps=1e-3):
      column_sums = torch.sum(prob[0], dim=0).data
      good_protos = column_sums > eps
      idxs = torch.nonzero(good_protos).squeeze()
      return tensor_proto[:, idxs, :], radii[:, idxs], targets[idxs]

def assign_cluster_radii_limited(cluster_centers, data, radii, target_labels):
    logits = compute_logits_radii(cluster_centers, data, radii) # [B, N, K]
    # print(logits.size())
    class_logits = (torch.min(logits).data-100)*torch.ones(logits.data.size()).cuda()
    target_labels = target_labels.unsqueeze(0)
    class_logits[target_labels] = logits.data[target_labels]
    logits_shape = logits.size()
    bsize = logits_shape[0]
    ndata = logits_shape[1]
    ncluster = logits_shape[2]
    prob = F.softmax(Variable(class_logits), dim=-1)
    return prob

def compute_logits_radii(cluster_centers, data, radii, prior_weight=1.):
    cluster_centers = torch.unsqueeze(cluster_centers, 1)   # [B, 1, K, D]
    data = torch.unsqueeze(data, 2)  # [B, N, 1, D]
    dim = data.size()[-1]
    radii = torch.unsqueeze(radii, 1)  # [B, 1, K]
    neg_dist = -torch.sum((data - cluster_centers)**2, dim=3)   # [B, N, K]

    logits = neg_dist / 2.0 / (radii)
    norm_constant = 0.5*dim*(torch.log(radii) + np.log(2*np.pi))

    logits = logits - norm_constant
    return logits

def imp_loss(logits, targets, labels):
        targets = targets.cuda()
        # determine index of closest in-class prototype for each query
        target_logits = torch.ones_like(logits.data) * float('-Inf')

        target_logits[targets] = logits.data[targets]
        _, best_targets = torch.max(target_logits, dim=1)

        # mask out everything...
        weights = torch.zeros_like(logits.data)

        # ...then include the closest prototype in each class and unlabeled)
        unique_labels = np.unique(labels.cpu().numpy())

        for l in unique_labels:

            class_mask = labels == l
            class_logits = torch.ones_like(logits.data) * float('-Inf')
            class_mask = class_mask.repeat(logits.size(0), 1)

            class_logits[class_mask] = logits[class_mask].data.reshape(-1)
            _, best_in_class = torch.max(class_logits, dim=1)

            weights[range(0, targets.size(0)), best_in_class] = 1.

        loss = weighted_loss(logits, Variable(best_targets), Variable(weights))
        return loss.mean()

def log_sum_exp(value, weights, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(weights*torch.exp(value0),
                                       dim=dim, keepdim=keepdim))

def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)

def weighted_loss(logits, targets, weights):
    logsumexp = log_sum_exp(logits, weights, dim=1, keepdim=False)
    loss_by_class = -1*class_select(logits,targets) + logsumexp
    return loss_by_class

"""## **Model**"""

torch.cuda.empty_cache()

"""**Change the architecture below**"""

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)

def load_Imp_conv(**kwargs):

  x_dim = kwargs['x_dim']
  hid_dim = kwargs['hid_dim']
  z_dim = kwargs['z_dim']

  def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )

  encoder = nn.Sequential(
    conv_block(x_dim[0], hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, z_dim),
    Flatten(),
    nn.Linear(z_dim * 8 * 8, 768)  # Adjust the input size of the linear layer based on the last conv layer's output
    )
  return Imp(encoder)

class Imp(nn.Module):
  def __init__(self, encoder):
    super(Imp, self).__init__()
    self.encoder = encoder.cuda()

  def set_forward_loss(self, sample):
    #prototypical network code
    sample_images = sample['images'].cuda()
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    n_labels = sample['n_label']
    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]

    #imp
    num_cluster_steps = 1
    init_sigma_l = 5
    log_sigma_l = torch.log(torch.FloatTensor([init_sigma_l]))
    log_sigma_l = Variable(log_sigma_l, requires_grad=True).cuda()

    #prior probabilities
    #for signatures... y_train values are to be mapped to nlabels
    #y_train = [1, images, nclusters]
    y_train = []
    for i in range(n_way):
        for j in range(n_support):
            y_train.append(i)
    y_train = torch.tensor(y_train).cuda()
    y_train = torch.unsqueeze(y_train, dim=0)

    y_test = []
    for i in range(n_way):
        for j in range(n_query):
            y_test.append(i)
    y_test = torch.tensor(y_test).cuda()
    # print("y_test: ",y_test)
    y_test = torch.unsqueeze(y_test, dim=0)

    nClusters = len(np.unique(y_train.data.detach().cpu().numpy()))
    nInitialClusters = nClusters

    #prob_train = [1, images, nclusters] --- only for training --- support set
    prob_train = one_hot(y_train, nClusters).cuda()
    #print(prob_train.size())
    prob_test = one_hot(y_test, nClusters).cuda()
    prob_test = torch.squeeze(prob_test, dim=0).detach().cpu().numpy()
    bsize = 1
    radii = Variable(torch.ones(bsize, nClusters)).cuda() * torch.exp(log_sigma_l)
    support_labels = torch.arange(0, nClusters).cuda().long()

    #encode images of the support and the query set
    #h_train = [1, images, encodings] --- support_set, h_test = [1, images, encodings] --- query_set

    h_train_images = x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]).clone().detach().requires_grad_(True)
    h_test_images = x_query.contiguous().view(n_way * n_query, *x_query.size()[2:]).clone().detach().requires_grad_(True)

    h_train = self.encoder.forward(h_train_images)
    h_train = torch.unsqueeze(h_train, dim=0)
    #print(h_train.size())
    h_test = self.encoder.forward(h_test_images)
    h_test = torch.unsqueeze(h_test, dim=0)

    h_train_dim = h_train.size(-1)
    h_test_dim = h_train.size(-1)

    #compute prototypes using h_train and prob_train
    #protos = [1, nclusters, z_dim]
    protos = compute_protos(h_train, prob_train)
    #print(protos.size())
    #calculate lambda --- threshold
    #estimate_lamda
    threshold = estimate_lamda(protos)


    #clustering loop
    for ii in range(num_cluster_steps):
      tensor_proto = protos.data
      for i, ex in enumerate(h_train[0]):
        idxs = torch.nonzero(y_train.data[0, i] == support_labels)[0]
        distances = compute_distances(tensor_proto[:, idxs, :], ex.data)
        if (torch.min(distances) > threshold):
          temp = np.random.binomial(size=10, n=1, p= 0.4)
          t = np.mean(temp)
          if (t > 0.5):
              nClusters, tensor_proto, radii = add_cluster(nClusters, tensor_proto, radii, ex=ex.data)
              support_labels = torch.cat([support_labels, y_train[0, i].data.view(1)], dim=0)

      #perform partial reassignment based on newly created labeled clusters
      if nClusters > nInitialClusters:
        support_targets = y_train.data[0, :, None] == support_labels
        prob_train = assign_cluster_radii_limited(Variable(tensor_proto), h_train, radii, support_targets)

      protos = Variable(tensor_proto).cuda()
      protos = compute_protos(h_train, Variable(prob_train.data, requires_grad=False).cuda())
      protos, radii, support_labels = delete_empty_clusters(protos, prob_train, radii, support_labels)
      # print("protos: ",protos.size())
      # print("support_labels: ",support_labels.size())
      # print("n_labels: ",len(n_labels))

    # print("support: ", support_labels,"\nsupport: ", support_labels.size(),"\n",)
    # print("protos: ", protos.size())
    # print("h_test: ", h_test.size())
    logits = compute_logits_radii(protos, h_test, radii).squeeze()
    # print("logits: ", logits.size())

    # print("n_labels: ",n_labels)
    #new loss
    '''
    temp = []
    df = pd.read_csv("/home/maharathy1/MTP/implementation/MIA/embeddings/chexpert/Sign_7-dmis-lab-biobert-v1_1_mean.csv")
    for i in support_labels:
      j = n_labels[i]
      loc = df.loc[(df['Abnormality_Label'] == j)]
      loc = loc.values.tolist()
      temp.append(loc[0][1:])
    signatures = np.array(temp)
    signatures = torch.tensor(signatures).cuda()
    signatures = torch.unsqueeze(signatures, dim=0)
    new_loss = torch.cdist(protos.float(), signatures.float(), p=2)
    diag = torch.diagonal(new_loss, 0)
    sign_loss = torch.mean(diag)
    '''

    #convert class targets into indicators for supports in each class
    labels = y_test.data
    labels[labels >= nInitialClusters] = -1
    support_targets = labels[0, :, None] == support_labels
    ce_loss = imp_loss(logits, support_targets, support_labels)
    # print("Sign Loss: ", sign_loss)
    # print("CE Loss: ", ce_loss)
    loss = ce_loss
    # print("loss: ", loss)

    #map support predictions back into classes to check accuracy
    _, support_preds = torch.max(logits.data, dim=1)
    softmax_layer = nn.Softmax(dim=1)
    r = softmax_layer(logits.data)
    y_pred = support_labels[support_preds]
    acc_val = torch.eq(y_pred, labels[0]).float().mean()

    return loss, {
        'loss': loss.item(),
        'acc': acc_val,
        'logits': logits.tolist()[0]
     }

  def set_forward_loss_1(self, sample):
    #prototypical network code
    sample_images = sample['images'].cuda()
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    n_labels = sample['n_label']
    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]

    #imp
    num_cluster_steps = 1
    init_sigma_l = 5
    log_sigma_l = torch.log(torch.FloatTensor([init_sigma_l]))
    log_sigma_l = Variable(log_sigma_l, requires_grad=True).cuda()

    #prior probabilities
    #for signatures... y_train values are to be mapped to nlabels
    #y_train = [1, images, nclusters]
    y_train = []
    for i in range(n_way):
        for j in range(n_support):
            y_train.append(i)
    y_train = torch.tensor(y_train).cuda()
    y_train = torch.unsqueeze(y_train, dim=0)

    y_test = []
    for i in range(n_way):
        for j in range(n_query):
            y_test.append(i)
    y_test = torch.tensor(y_test).cuda()
    # print("y_test: ",y_test)
    y_test = torch.unsqueeze(y_test, dim=0)

    nClusters = len(np.unique(y_train.data.detach().cpu().numpy()))
    nInitialClusters = nClusters

    #prob_train = [1, images, nclusters] --- only for training --- support set
    prob_train = one_hot(y_train, nClusters).cuda()
    #print(prob_train.size())
    prob_test = one_hot(y_test, nClusters).cuda()
    prob_test = torch.squeeze(prob_test, dim=0).detach().cpu().numpy()
    bsize = 1
    radii = Variable(torch.ones(bsize, nClusters)).cuda() * torch.exp(log_sigma_l)
    support_labels = torch.arange(0, nClusters).cuda().long()

    #encode images of the support and the query set
    #h_train = [1, images, encodings] --- support_set, h_test = [1, images, encodings] --- query_set

    h_train_images = x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]).clone().detach().requires_grad_(True)
    h_test_images = x_query.contiguous().view(n_way * n_query, *x_query.size()[2:]).clone().detach().requires_grad_(True)

    h_train = self.encoder.forward(h_train_images)
    h_train = torch.unsqueeze(h_train, dim=0)
    #print(h_train.size())
    h_test = self.encoder.forward(h_test_images)
    h_test = torch.unsqueeze(h_test, dim=0)

    h_train_dim = h_train.size(-1)
    h_test_dim = h_train.size(-1)

    #compute prototypes using h_train and prob_train
    #protos = [1, nclusters, z_dim]
    protos = compute_protos(h_train, prob_train)
    #print(protos.size())
    #calculate lambda --- threshold
    #estimate_lamda
    threshold = estimate_lamda(protos)


    #clustering loop
    for ii in range(num_cluster_steps):
      tensor_proto = protos.data
      for i, ex in enumerate(h_train[0]):
        idxs = torch.nonzero(y_train.data[0, i] == support_labels)[0]
        distances = compute_distances(tensor_proto[:, idxs, :], ex.data)
        if (torch.min(distances) > threshold):
          temp = np.random.binomial(size=10, n=1, p= 0.4)
          t = np.mean(temp)
          if (t > 0.5):
              nClusters, tensor_proto, radii = add_cluster(nClusters, tensor_proto, radii, ex=ex.data)
              support_labels = torch.cat([support_labels, y_train[0, i].data.view(1)], dim=0)

      #perform partial reassignment based on newly created labeled clusters
      if nClusters > nInitialClusters:
        support_targets = y_train.data[0, :, None] == support_labels
        prob_train = assign_cluster_radii_limited(Variable(tensor_proto), h_train, radii, support_targets)

      protos = Variable(tensor_proto).cuda()
      protos = compute_protos(h_train, Variable(prob_train.data, requires_grad=False).cuda())
      protos, radii, support_labels = delete_empty_clusters(protos, prob_train, radii, support_labels)
      # print("protos: ",protos.size())
      # print("support_labels: ",support_labels.size())
      # print("n_labels: ",len(n_labels))

    # print("support: ", support_labels,"\nsupport: ", support_labels.size(),"\n",)
    # print("protos: ", protos.size())
    # print("h_test: ", h_test.size())
    logits = compute_logits_radii(protos, h_test, radii).squeeze()
    # print("logits: ", logits.size())

    # print("n_labels: ",n_labels)
    #new loss
    '''
    temp = []
    df = pd.read_csv("/home/maharathy1/MTP/implementation/MIA/embeddings/chexpert/Sign_7-dmis-lab-biobert-v1_1_mean.csv")
    for i in support_labels:
      j = n_labels[i]
      loc = df.loc[(df['Abnormality_Label'] == j)]
      loc = loc.values.tolist()
      temp.append(loc[0][1:])
    signatures = np.array(temp)
    signatures = torch.tensor(signatures).cuda()
    signatures = torch.unsqueeze(signatures, dim=0)
    new_loss = torch.cdist(protos.float(), signatures.float(), p=2)
    diag = torch.diagonal(new_loss, 0)
    sign_loss = torch.mean(diag)
    '''

    #convert class targets into indicators for supports in each class
    labels = y_test.data
    labels[labels >= nInitialClusters] = -1
    support_targets = labels[0, :, None] == support_labels
    ce_loss = imp_loss(logits, support_targets, support_labels)
    loss = ce_loss

    #map support predictions back into classes to check accuracy
    _, support_preds = torch.max(logits.data, dim=1)
    softmax_layer = nn.Softmax(dim=1)
    r = softmax_layer(logits.data)
    y_pred = support_labels[support_preds]
    acc_val = torch.eq(y_pred, labels[0]).float().mean()

    #metrics
    gt = torch.squeeze(y_test, dim=0).detach().cpu().numpy()
    gt_list = list(gt)
    c = len(gt_list)
    i = c*n_query
    pr = y_pred.detach().cpu().numpy()
    support = list(support_labels.detach().cpu().numpy())
    r = r.detach().cpu().numpy()
    # print("r: ", r)
    prob = [[0.0]*c]*i
    prob = np.array(prob)
    # print("prob: ", prob)
    # print(type(prob), type(r))
    for v in range(i):
      for k in gt_list:
        for j in range(len(support)):
          if support[j] == k:
            if prob[v][k] < r[v][j]:
              # print("true: ",r[v][j])
              prob[v][k] = np.copy(r[v][j])
    # print("prob after: ", prob)
    return loss, {
        'loss': loss.item(),
        'acc': acc_val,
        'logits': logits.tolist()[0],
        'y_test': prob_test,
        'y_pred':prob,
        'gt': gt,
        'pr': pr
    }

"""## **Training**"""

from tqdm import tqdm_notebook
from tqdm import trange

def train(eps, model, optimizer, train_x, train_y, val_x, val_y, n_way, n_support, n_query, max_epoch, epoch_size, temp_str):
#def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
  #divide the learning rate by 2 at each epoch, as suggested in paper
  scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
  epoch = 0 #epochs done so far
  stop = False #status to know when to stop
  prev_val_acc = 0
  prev_val_loss = 1000
  flag = 0
#   train_loss = []
#   train_acc = []
#   val_loss = []
#   val_acc = []
  while flag != 1:

    running_loss = 0.0
    running_acc = 0.0
    for episode in range(epoch_size):
      sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
      optimizer.zero_grad()
      loss, output = model.set_forward_loss(sample)
      running_loss += output['loss']
      running_acc += output['acc']
      loss.backward()
      optimizer.step()
    epoch_loss = running_loss / epoch_size
    epoch_acc = running_acc / epoch_size
#     print(type(epoch_loss))
    print('Epoch {:d} Train -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc))
    epoch += 1
#     train_loss.append(epoch_loss)
#     train_acc.append(epoch_acc)
#     print(train_loss, train_acc)
    scheduler.step()

    val_loss = 0.0
    val_acc = 0.0
    for episode in range(epoch_size):
      sample = extract_sample(3, 5, 5, val_x, val_y)
      valdiation_loss, val_output = model.set_forward_loss(sample)
      val_loss += val_output['loss']
      val_acc += val_output['acc']
    avg_loss = val_loss / epoch_size
    avg_acc = val_acc / epoch_size

    if epoch == 1:
          prev_val_acc = avg_acc
          prev_val_loss = avg_loss

    if epoch > 1:
        loss_diff = prev_val_loss - avg_loss
        loss_diff = abs(loss_diff)
        if round(loss_diff, 3) < eps or round(loss_diff, 3) == eps:
            if epoch > 5 or epoch == 5:
                torch.save(model, temp_str + '.pth')
#                 torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),
#                             'optimizer_state_dict': optimizer.state_dict(),'loss': avg_loss,}
#                              ,'best_model_path_' + temp_str + '_dict.pth')
                flag = 1
        print('Prev Val Results -- Loss: {:.4f} Acc: {:.4f}'.format(prev_val_loss, prev_val_acc))
        print(loss_diff)
        prev_val_acc = avg_acc
        prev_val_loss = avg_loss

    print('Val Results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
    print("\n")
#     print(type(avg_loss))
#     val_loss.append(avg_loss)
#     val_acc.append(avg_acc)
#     print(val_loss, val_acc)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score

"""## **Testing**"""

def test(temp_str, test_x, test_y, n_way, n_support, n_query, test_episode, lock, test_ls):
  gt=[]
  pr=[]
  running_loss = 0.0
  running_acc = 0.0
  model = torch.load(temp_str + '.pth')
#   model = torch.load(temp_str + '.pth')
  for episode in range(test_episode):
    sample = extract_sample_test(test_ls, n_way, n_support, n_query, test_x, test_y)
#     print("0:", test_ls)
#     print("1: ", sample['n_label'])
#     display_sample(sample_example['images'])
    loss, output = model.set_forward_loss_1(sample)
    running_loss += output['loss']
    running_acc += output['acc']
    for i in output['gt']:
      gt.append(i)
    for i in output['pr']:
      pr.append(i)
#     print("gt: ", gt, "\npr: ", pr)
    if lock == 1:
      y_test = output['y_test']
      prob = output['y_pred']
      lock = False
    else:
      y_test = np.concatenate((y_test, output['y_test']), axis=0)
      prob = np.concatenate((prob, output['y_pred']), axis=0)

  avg_loss = running_loss / test_episode
  avg_acc = running_acc / test_episode
  print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
  precision, recall, fscore, support = score(gt, pr)
  print('precision: {}'.format(precision))
  print('recall: {}'.format(recall))
  print('fscore: {}'.format(fscore))
  print('support: {}'.format(support))
  # print("gt: ", gt, "\npr: ", pr)
  # print("y_test: ", y_test,"\nprob: ", prob)
  # print("y_test: ", y_test.shape(),"\nprob: ", prob.shape())
  auroc_values = []
  for i in range(n_way):
      auroc_values.append(roc_auc_score(y_test[:,i], prob[:,i], multi_class='ovo'))
  c = 0
  for i in test_ls:
    print("AUROC for ",i,": ", auroc_values[c])
    c += 1

  return auroc_values[0], auroc_values[1], auroc_values[2], precision, recall, fscore

path = "//home/maharathy1/MTP/implementation/MIA/models/ablation_1/chexpert_128/split1/"

for i in range(10):
    print("Model: ", i + 1)
    model = load_Imp_conv(
        x_dim=(3,128,128),
        hid_dim=128,
        z_dim=768,
        )
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    n_way = 8
    n_support = 5
    n_query = 5
    X = X_train
    y = y_train
    X_val = X_val
    y_val = y_val
    max_epoch = 2
    epoch_size = 500
    temp_str = path+'ablation_1_chexpert_split1_128_' + str(i)
    train(0.005, model, optimizer, X, y, X_val,y_val, n_way, n_support, n_query, max_epoch, epoch_size, temp_str)

"""## **Model**"""

test_labels = np.unique(y_test)
print(type(test_labels))
test_labels = test_labels.tolist()
print(type(test_labels))
print(test_labels)

list_1 = []
list_2 = []
list_3 = []
precision = []
recall = []
f1score = []
for j in range(10):
    print("Best Model: ", j + 1)
    ab_1 = 0
    ab_2 = 0
    ab_3 = 0
    p = []
    r = []
    f = []
    n_way = 3
    n_support = 5
    n_query = 1
    test_x = X_test
    test_y = y_test
    test_episode = 1000
    lock = 1
    temp_str = path+'ablation_1_chexpert_split1_128_' + str(j)
    ab_1, ab_2, ab_3, p, r, f = test(temp_str, test_x, test_y, n_way, n_support, n_query, test_episode, lock, test_labels)
    print(test_labels[0],": ", ab_1)
    print(test_labels[1],": ", ab_2)
    print(test_labels[2],": ", ab_3)
    print("\n")
    list_1.append(ab_1)
    list_2.append(ab_2)
    list_3.append(ab_3)
    precision.append(p)
    recall.append(r)
    f1score.append(f)
arr_1 = np.array(list_1)
arr_2 = np.array(list_2)
arr_3 = np.array(list_3)
pre = np.array(precision)
re = np.array(recall)
f1 = np.array(f1score)

mean_pre = np.round(np.mean(pre, axis=0), decimals=4)
mean_re = np.round(np.mean(re, axis=0), decimals=4)
mean_f1 = np.round(np.mean(f1, axis=0), decimals=4)
std_pre = np.round(np.std(pre, axis=0), decimals=4)
std_re = np.round(np.std(re, axis=0), decimals=4)
std_f1 = np.round(np.std(f1, axis=0), decimals=4)

mean_auroc = []
mean_auroc.append(np.round(np.mean(arr_1),4))
mean_auroc.append(np.round(np.mean(arr_2),4))
mean_auroc.append(np.round(np.mean(arr_3),4))
std_auroc = []
std_auroc.append(np.round(np.std(arr_1),4))
std_auroc.append(np.round(np.std(arr_2),4))
std_auroc.append(np.round(np.std(arr_3),4))

print("Mean AUROC of ",test_labels[0],": ", mean_auroc[0],"\n Std AUROC of ",test_labels[0],":", std_auroc[0])
print("Mean AUROC of ",test_labels[1],": ", mean_auroc[1],"\n Std AUROC of ",test_labels[1],":", std_auroc[1])
print("Mean AUROC of ",test_labels[2],": ", mean_auroc[2],"\n Std AUROC of ",test_labels[2],":", std_auroc[2])

data = {
    'mean_pre': mean_pre,
    'std_pre': std_pre,
    'mean_re': mean_re,
    'std_re': std_re,
    'mean_f1': mean_f1,
    'std_f1': std_f1,
    'mean_auroc': mean_auroc,
    'std_auroc': std_auroc
}

# Create the dataframe
df = pd.DataFrame(data, index=test_labels)

df.to_csv('/home/maharathy1/MTP/implementation/MIA/models/ablation_1/chexpert_128/csvs/output_chexpert_ablation_1_split1_128_final.csv')
