"""
Extract features for visualization
"""
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from OrdinalEntropy import ordinal_entropy
import scipy.io as scio
from models import MLP, MLP_classification
import cv2
from sklearn.manifold import TSNE
import torch.nn.functional as F
import random


def main(regression=True, oe=True):
    m = 100
    lr = 1e-3
    epochs = 20000
    dataset_train = "train.npz"
    dataset_test = "test.npz"
    Lambda_d = 1e-3

    if regression:
        model = MLP(m).cuda()
        loss_function = nn.MSELoss().cuda()
        description = 'regression'
        if oe:
            description = description + '+OrdinalEntropy'
    else:
        num_bins= 10
        model = MLP_classification(m=m, bins=num_bins).cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
        description = 'classification'
        softmax = nn.Softmax()


    d = np.load(dataset_train)
    X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
    d = np.load(dataset_test)
    X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]

    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)

    upper = np.percentile(y_train, 90)
    upper = np.where(y_train<upper)[0]
    X_train = X_train[upper]
    y_train = y_train[upper]
    down = np.percentile(y_train, 10)
    down = np.where(y_train>down)[0]
    X_train = X_train[down]
    y_train = y_train[down]

    X_train = Variable(torch.from_numpy(X_train), requires_grad=True).float().cuda()
    y_train = Variable(torch.from_numpy(y_train), requires_grad=True).float().cuda()
    X_test = Variable(torch.from_numpy(X_test), requires_grad=True).float().cuda()
    y_test = Variable(torch.from_numpy(y_test), requires_grad=True).float().cuda()


    """
    Train models
    """
    print('Start training')
    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        if epoch % 1000 == 0:
            print(epoch)
        model.train()
        optimizer.zero_grad()
        pred, feature = model(X_train)
        if regression:
            loss = loss_function(pred, y_train)
            if oe:
                loss_oe = ordinal_entropy(feature, y_train) * Lambda_d
            else:
                loss_oe = loss * 0
            loss_all = loss + loss_oe
        else:
            _max = torch.max(y_train)
            _min = torch.min(y_train)
            _width = (_max - _min) / num_bins
            y_class = (y_train - _min) // _width
            y_class[y_class == num_bins] = num_bins - 1
            y_class = torch.squeeze(y_class)
            y_class = y_class.long()
            loss = loss_function(pred, y_class)
            loss_all = loss

        loss_all.backward()
        optimizer.step()

    print('Training Finished')

    """
    Extract features from the test set
    """
    model.eval()
    with torch.no_grad():
        pred, features = model(X_test)
        features = features.cpu().data.numpy()

        samples = random.sample(range(0, len(y_test)-1), 3000)  # random sample 3000 features to visualization
        features = features[samples]
        y_test = y_test[samples]
        pred = pred[samples]

        print('Start Embedding')
        ts = TSNE(n_components=3)
        ts.fit_transform(features)
        output = ts.embedding_
        output = torch.tensor(output)
        output = F.normalize(output)
        output = output.numpy()

        y_test = y_test.cpu().numpy()

        if regression:
            pred = pred.cpu().numpy()
        else:
            pred = torch.argmax(softmax(pred), 1, keepdim=True)
            pred = pred.float()
            pred = _min + _width * pred
            pred = pred.cpu().numpy()

        save_points(output, y_test, pred,
                    description + '.mat')
        print('Save embeddings!')


def save_points(embeds, labels, preds, path):
    train_dict = {}
    train_dict['embeds'] = embeds
    train_dict['labels'] = labels
    train_dict['preds'] = preds
    scio.savemat(path, train_dict)


if __name__ == "__main__":
    regression = True  # choose the regression/classification model, i.e. True=regression, False=classification
    oe = True  # use the ordinal entropy or not, i.e. True = use, False = do not use

    main(regression, oe)