# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:31:26 2025

@author: Wang Jialiang
"""

import os
from tqdm import tqdm
import numpy as np
from numpy import linalg
import pandas as pd
rr = os.path.dirname(os.path.abspath( __file__ ) )
rng = np.random.default_rng()

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def loss(y_hat, label, weights, lambda_ = 100, yetta_ = 0.1):
    
    cross_entropy = y_hat - np.max(y_hat, axis=1, keepdims=True)
    cross_entropy = cross_entropy - np.log(np.sum(np.exp(cross_entropy), axis=1, keepdims=True))
    cross_entropy = -np.sum(label * cross_entropy) / label.shape[0]
    
    l2_reg = lambda_ * np.sum(np.maximum(0, -weights)**2)
    l1_reg = yetta_* np.sum(np.abs(weights))
    
    
    return cross_entropy + l1_reg + l2_reg

def dloss(y_hat, labels, weights, hidden, lambda_ = 100, yetta_ = 0.1):
    
    derro = (np.exp(y_hat - np.max(y_hat, axis = 1, keepdims = True)) / np.sum(np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True)), axis = 1, keepdims = True)) - labels
    derro /= y_hat.shape[0]
    grad_w = np.dot(hidden.T, derro)
    
    grad_w += yetta_ * np.sign(weights)
    grad_w -= lambda_ * 2 * np.maximum(0, -D_mk)
    
    return grad_w

dataset = np.load(rr + '\\Dataset\\spec_gen.npy')
labels = np.load(rr + '\\Dataset\\spec_label.npy')

R_space = pd.read_csv(rr + '\\Response_Matrix\\file_name.csv', index_col = 0)
R_lm = R_space.to_numpy()
I_nm = np.dot(dataset, R_lm)
D_mk = np.abs(rng.standard_normal((40, 3))) * np.sqrt(2 / 40)

sparsity = 0.1 # control network sparsity
lr = 0.002
n_epoch = 30
n_batch = 20
bat_size = 30
# bat_size = len(labels) // n_batch

loss_hist = np.zeros(n_epoch)
out_hist = np.zeros((n_epoch, 3))
accuracy_hist = np.zeros(n_epoch)

# main
for epoch in range(n_epoch):
    
    print('\n Start #'+str(epoch)+' epoch training')
    # Shuffle the training set
    perm = rng.permutation(len(labels))
    labels_train = labels[perm]
    input_neurons = I_nm[perm]
    
    # Train on mini-batches
    print('\n Mini batch progressing... \n')
    for bat in tqdm(range(n_batch)):
        # Select a mini-batch
        start = bat * bat_size
        end = start + bat_size
        batch_inputs = input_neurons[start:end]
        batch_labels = labels_train[start:end]
        # forward propagation
        output_neurons = np.dot(batch_inputs, D_mk)
        # loss function
        n_classes = 3
        bat_labels_onehot = np.eye(3)[batch_labels]
        loss_bat = loss(output_neurons, bat_labels_onehot, D_mk)
        # backward propagation via coordinate descent
        for i in range(D_mk.shape[0]):
            
            for j in range(D_mk.shape[1]):
                
                D_ij_old = D_mk[i, j]
                grad_out = dloss(output_neurons, bat_labels_onehot, D_mk, batch_inputs)
                grad_ij = grad_out[i, j]
                
                D_mk[i, j] -= lr * grad_ij
                if abs(D_mk[i, j]) < sparsity:
                    D_mk[i, j] = 0
        
    # training log
    loss_hist[epoch] = loss_bat
    # evalutate accuracy
    output_neurons = np.dot(I_nm, D_mk)
    prob = softmax(output_neurons)
    significance = np.max(prob, axis = 1) > 0.5
    predicted_labels = np.argmax(output_neurons, axis=1)
    predict_results = predicted_labels == labels
    predict_results = predict_results * significance
    accuracy_hist[epoch] = np.mean(predict_results.astype(int) / 1)
    R_recs = np.dot(R_lm, D_mk)
    
np.save(rr + '\\Models\\spectral_classifier.npy', D_mk)
np.save(rr + '\\Models\\R_classifier.npy', R_recs)
R_recs = pd.DataFrame(R_recs)
R_recs.to_csv(rr + '\\Models\\R_classifier.csv')

accuracy_hist = pd.DataFrame(accuracy_hist)
accuracy_hist.to_csv(rr + '\\Models\\accuracy_hist.csv')
    


    
        