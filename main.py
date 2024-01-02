import math
import numpy as np
import pandas
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from torch import nn as nn


def get_model(n_inputs,device = 'cpu'):
    model = nn.Sequential(
            nn.Linear(n_inputs,1000),
            nn.ReLU(),
            nn.Linear(1000,2),
            nn.SELU()
            )
    model.to(device)
    return model

def echo(string, padding=80):
    padding = " " * (padding - len(string)) if padding else ""
    print(string + padding, end='\r')

def train_model(model, input_train,output_train,device = 'cpu', Epocs=10,lr=0.7, weights = [1,1],val_split=0.3, l2_lambda=0, momentum = 0.9):
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights).float())
    input_train = torch.from_numpy(input_train).float()
    output_train = torch.from_numpy(np.squeeze(output_train)).long()
    
    I=torch.randperm(np.shape(input_train)[0])
    n=int(np.round(val_split*np.shape(input_train)[0]))
    val_input_train = input_train[I[:n]].to(device)
    val_output_train = output_train[I[:n]].to(device)
    
    
    input_train = input_train[I[n:]].to(device)
    output_train = output_train[I[n:]].to(device)
    
    min_train_val_loss=np.inf
    # move the tensor to gpu
    optimizer=optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # optimizer=optim.Adam(model.parameters(), lr=lr/10)
    # optimizer=optim.Adagrad(model.parameters(), lr=lr)
    # optimizer=optim.Adadelta(model.parameters(), lr=lr)
    # optimizer=optim.Adamax(model.parameters(), lr=lr/20)
    # optimizer=optim.RMSprop(model.parameters(), lr=lr/10)
    
    optimizer.zero_grad()
    model_log=[]

    # train the model for 10 epochs
    for epoch in range(Epocs):
        # forward pass
        train_pred = model(input_train)
        # compute loss

        loss = loss_fn(train_pred, output_train)
        
        
        if l2_lambda != 0:
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        val_pred = model(val_input_train)
        val_loss=loss_fn(val_pred, val_output_train)
        if val_loss < min_train_val_loss:
            min_train_val_loss=val_loss
            model_log=model
            
        if val_loss > 1 and epoch > 100:
            print(f'Пришлось прерваться на эпохе: {epoch}')
            break
        if np.mod(epoch,10)==0:
            message = f'{epoch}:      {loss}      {val_loss}     {min_train_val_loss}'
            echo(message)
            model = model_log # dubeous
    
    message = f'{epoch}:      {loss}      {val_loss}     {min_train_val_loss}'
    print(message)
    
    
    train_pred=np.argmax(model(val_input_train).detach().cpu().numpy(),1)
    output_train=val_output_train.detach().cpu().numpy()
    err=output_train.squeeze()-train_pred.squeeze()
    acc=1-np.mean(np.abs(err))
    err=np.sum(err[err<0])
    print(f'Минимальные потери: {min_train_val_loss}')
    print(f'Accuracy = {acc*100} %')
    print(f'Всего выживших: {np.sum(output_train)}')
    print(f'Распознанные выжившие: {np.sum(train_pred)}')
    print(f'Было распознано ошибочно: {np.abs(err)}\n')
    input_train.detach()
    return model_log


def get_prediction(model,input,device = 'cpu'):
    input = torch.from_numpy(input).float().to(device)
    return np.argmax(model(input).detach().cpu().numpy(),axis=1)

def split_data(Path, input_tag, output_tag = None):
    data = pandas.read_csv(Path)

    input = (data[input_tag].to_numpy())
    
    if output_tag is not None:
        output = (data[output_tag].to_numpy())
    else:
        output = None
        
    return input, output




def preprocess_data(all_data, pca = None, mean_ = None, norm = None):
    # male/female to 0/1
    all_data[:,2] = [int(sex == 'female') for sex in all_data[:,2]]

    # Entry station to 0-2
    all_data[all_data[:,8] == 'S',8] = 0
    all_data[all_data[:,8] == 'C',8] = 1
    all_data[all_data[:,8] == 'Q',8] = 2

    all_data = np.column_stack((all_data,[math.isnan(age) for age in all_data[:,3]] and [math.isnan(entry) for entry in all_data[:,8]]))
    # all_data = np.column_stack((all_data,[math.isnan(entry) for entry in all_data[:,8]]))
    # all_data = np.column_stack((all_data,[math.isnan(age) for age in all_data[:,3]]))



    # Cabin data to 1 if data exists, 0 otherwise
    all_data[:,7] = [int(cabin is np.nan) for cabin in all_data[:,7]]
    
    all_data[[math.isnan(entry) for entry in all_data[:,6]],6] = np.nanmean(all_data[:,6])
    
    # for entry in all_data:
    #     all_data[math.isnan(entry[6]),6] = np.nanmean(all_data[all_data[:,1] == entry[1],6])
    
    all_data[[math.isnan(entry) for entry in all_data[:,8]],8] = np.nanmean(all_data[:,8])


    # If the age  is nan replace it with mean age of passengers with the same sex ang ticket class
    all_data[[math.isnan(age) for age in all_data[:,3]],3] = [np.nanmean(all_data[np.squeeze(np.all([[all_data[:,0] == entry[0]] , [all_data[:,2] == entry[2]]],axis = 0)),3]) for entry in all_data if math.isnan(entry[3])]


    input = np.delete(all_data,1,axis = 1).astype(np.float64)
    
    if mean_ is None:
        mean_ = np.mean(input,axis = 0)
        
        input = input - mean_
    
    if norm is None:
        norm = np.mean(np.abs(input),axis = 0)
        
        input = input/norm
    
    if pca == None:
        pca = PCA(n_components=np.shape(input)[1])
        pca.fit(input)
        
    signal_transformed=pca.transform(input)
    input=signal_transformed
    return input, norm,mean_,pca


all_data, output = split_data(Path = 'Data/train.csv',
                              input_tag = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'], 
                              output_tag = ['Survived'])


input, norm,mean_,pca = preprocess_data(all_data)

model = get_model(n_inputs = np.shape(input)[1])

model = train_model(model,
                    input,
                    output,
                    Epocs=30000, 
                    lr = 0.0001,
                    weights=[0.38,1], 
                    val_split=0.3,
                    l2_lambda=0.0000001,
                    momentum = 0.9,)
# weights=[0.38,1]


all_data_test,PID  = split_data(Path = 'Data/test.csv',
                              input_tag = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'], 
                              output_tag = ['PassengerId'])

input_test,_,_,_ = preprocess_data(all_data_test, pca = pca, mean_ = mean_, norm = norm)

results = get_prediction(model,input_test)

results = np.column_stack((np.squeeze(PID),np.squeeze(results)))

import pandas as pd 
df = pd.DataFrame(results)
df.to_csv("results.csv", header=['PassengerId','Survived'], index=False)
