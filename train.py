#!/usr/bin/env python3

'''
Quick Function to Train and Save models with new SolTranNet stuff
'''

import pandas as pd 
import torch
import numpy as np 
from transformer import make_model
from data_utils import load_data_from_df, construct_loader
import pickle, argparse, sys, os

parser=argparse.ArgumentParser(description='Trand SolTranNet on datasets.')
parser.add_argument('--trainfile',type=str,required=True,help='Absolute PATH to the training data file.')
parser.add_argument('--testfile',type=str,required=True,help='Absolute PATH to the testing data file.')
parser.add_argument('--datadir',type=str,default='weights/',help='Name of the Directory to save the weights and output predictions. Defaults to weights/')
parser.add_argument('--seed',type=int,required=True,help='Random Seed for model training.')
args=parser.parse_args()

#setting up the output file prefix
outprefix=f'{args.datadir}seed{args.seed}'

#loading the dataset
print('Trainfile:',args.trainfile)
print('Testfile:',args.testfile)
print('Loading train and test data')

torch.manual_seed(args.seed)
np.random.seed(args.seed)
batch_size=8
trainX, trainy=load_data_from_df(args.trainfile,add_dummy_node=True,one_hot_formal_charge=True)
data_loader=construct_loader(trainX,trainy,batch_size)
testX, testy=load_data_from_df(args.testfile,add_dummy_node=True,one_hot_formal_charge=True)
testdata_loader=construct_loader(testX,testy,batch_size,shuffle=False)

#setting up the model
d_atom=trainX[0][0].shape[1]

model_params= {
    'd_atom': d_atom,
    'd_model': 8,
    'N': 8,
    'h': 2,
    'N_dense': 1,
    'lambda_attention': 0.5,
    'leaky_relu_slope': 0.1,
    'dense_output_nonlinearity': 'relu',
    'dropout': 0.1,
    'aggregation_type': 'mean'
}

print('Making Model')
model=make_model(**model_params)
param_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters:',param_count)

#starting the training loop
model.train()
criterion=torch.nn.SmoothL1Loss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.04,momentum=0.06,weight_decay=0)

losses=[]
num_epochs_since_improvement=0
last_test_rmse=9999999999
last_test_r2=-1

print('Training')
iteration=0
for epoch in range(50000):
    for batch in data_loader:
        iteration+=1
        optimizer.zero_grad()
        adjacency_matrix, node_features, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        y_pred=model(node_features,batch_mask,adjacency_matrix,None)
        loss=criterion(y_pred,y)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2)
        optimizer.step()
    
    #end of 1 epoch -- time to evaluate
    model.eval()
    gold=np.array([])
    preds=np.array([])

    for t_batch in data_loader:
        t_adjacency_matrix, t_node_features, t_y = t_batch
        gold=np.append(gold,t_y.tolist())
        t_batch_mask = torch.sum(torch.abs(t_node_features), dim=-1) != 0
        t_y_pred=model(t_node_features,t_batch_mask,t_adjacency_matrix,None)
        preds=np.append(preds,t_y_pred.tolist())

    test_rmse=np.sqrt(np.mean((preds-gold)**2))
    test_r2=np.corrcoef(preds,gold)[0][1]**2
    model.train()

    if test_r2 > last_test_r2 or test_rmse < last_test_rmse:
        last_test_rmse=test_rmse
        last_test_r2=test_r2
        num_epochs_since_improvement=0
    else:
        num_epochs_since_improvement+=1

    #print the stats
    print(f'-----------------------------------------')
    print(f'Epoch: {epoch}/50000')
    print(f'Test RMSE: {test_rmse}')
    print(f'Test R2: {test_r2}')
    print(f'Num since Improvement: {num_epochs_since_improvement}')
    print(f'Best RMSE: {last_test_rmse}')
    print(f'Best R2: {last_test_r2}')
    print(f'----------------------------------')

    #now we check if dynamic stopping was signalled
    if num_epochs_since_improvement==4:
        print(f'Early termination signalled! Stopping training!\n{epoch}/50000 Completed.')

#Saving the resulting weights
if not os.path.isdir(args.datadir):
    os.mkdir(args.datadir)
print('Saving Model:',outprefix+'.weights')
torch.save(model.state_dict(),outprefix+'.weights')

#final evaluations
print('Final Evaluations!')
print('Training Set:')
model.eval()

gold=np.array([])
preds=np.array([])
for batch in data_loader:
    adjacency_matrix, node_features, y = batch
    batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
    y_pred=model(node_features,batch_mask,adjacency_matrix,None)
    gold=np.append(gold,y.tolist())
    preds=np.append(preds,y_pred.tolist())
r2=np.norrcoef(preds,gold)[0][1]**2
rmse=np.sqrt(np.mean((preds-gold)**2))
print('Training R2:',r2)
print('Training RMSE:',rmse)


print('Test Set:')
gold=np.array([])
preds=np.array([])
for batch in testdata_loader:
    adjacency_matrix, node_features, y = batch
    batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
    y_pred=model(node_features,batch_mask,adjacency_matrix,None)
    gold=np.append(gold,y.tolist())
    preds=np.append(preds,y_pred.tolist())
r2=np.norrcoef(preds,gold)[0][1]**2
rmse=np.sqrt(np.mean((preds-gold)**2))
print('Test R2:',r2)
print('Test RMSE:',rmse)

#dumping the predictions
test_df=pd.read_csv(args.testfile)
pcol=preds.tolist()
gcol=gold.tolist()
test_df['p']=pcol
test_df['g']=gcol
test_df.to_csv(outprefix+'.data')
