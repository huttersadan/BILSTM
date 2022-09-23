# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 10:15:56 2021

@author: yinghy
"""
import json

def start_end_position(labels):
    results = []
    status = 0
    for i in range(len(labels)):
        if labels[i] != 'O' and status == 0:
            start = i
            status = 1
        elif labels[i] != 'I' and status == 1:
            end = i-1
            status = 0
            results.append((start,end))
        
    if status == 1:
        results.append((start,len(labels)))
        
    return results
            


def f1_partial(y_true,y_pred,weight = 0.5):
    #left or right boundary match contributes to the TP sum with a weight
    true_sum = 0
    pred_sum = 0
    TP_sum = 0
    TP_par = 0
    
    #compute the four by traversing all the tags
    #relaxation: by seqeval, begin with 'I' is acceptable
    true_entities = start_end_position(y_true)
    pred_entities = start_end_position(y_pred)
    
    true_sum = len(true_entities)
    pred_sum = len(pred_entities)
    TP_sum = len(set(true_entities) & set(pred_entities))
    TP_left = len(set([k[0] for k in true_entities]) & set([k[0] for k in pred_entities]))
    TP_right = len(set([k[1] for k in true_entities]) & set([k[1] for k in pred_entities]))
    TP_par = TP_left + TP_right - 2*TP_sum
    
    print(TP_sum,TP_par,true_sum,pred_sum)
    #return F1:
    precision = (TP_sum + weight * TP_par)/pred_sum
    recall = (TP_sum + weight * TP_par)/true_sum
    f1 = 2 * precision * recall/(precision + recall)
    
    return precision,recall,f1

def f1_intersection(y_true,y_pred):
    #left O out and calculate the tag-level f1
    ytrue_transfer = []
    ypred_transfer = []
    for i in range(len(y_true)):
        if y_true[i] != 'O' or y_pred[i]!='O':
            ytrue_transfer.append(y_true[i] != 'O')
            ypred_transfer.append(y_pred[i] != 'O')
            
    TP = sum([ytrue_transfer[i] and ypred_transfer[i] for i in range(len(ytrue_transfer))])
    print(TP)
    
    precision = TP/sum(ypred_transfer)
    recall = TP/sum(ytrue_transfer)
    
    f1 = 2 * precision * recall/(precision + recall)
    
    return precision,recall,f1
    

if __name__ == '__main__':
    # y true and y pred both BIO tag sequence
    # we will calculate the overall f1, so we need to concatenate all the labels first
    
    #this is for BOND postprocess
    d = {0:'O',1:'B',2:'I'}
    
    pred_file = '/media/sdd1/luosx/NER/BOND/outputs/medmention-extra/parse_self_training/roberta_reinit0_begin900_period450_soft_hp5.9_3_1e-5/test_predictions.txt'
    true_file = '/media/sdd1/luosx/NER/BOND/dataset/medmention-extra-BIO/test.json'
    
    true_set = json.load(open(true_file))
    pred_set = []
    
    with open(pred_file) as f:
        for line in f:
            lb = eval(line.strip().split('\t')[-1])
            for j in range(len(lb)):
                if lb[j][0] == 'B':
                    lb[j] = 'B'
                elif lb[j][0] == 'I':
                    lb[j] = 'I'
                
            pred_set.append(lb)
    
    true_set = [[d[i] for i in k['tags']] for k in true_set]
    
    print(len(true_set))
    print(len(pred_set))
    
    
    no = []
    #compute F1
    assert len(true_set) == len(pred_set)
    for i in range(len(true_set)):
        if len(true_set[i])!=len(pred_set[i]):
            #print(true_set[i],len(true_set[i]))
            #print(pred_set[i],len(pred_set[i]))
            no.append(i)
        
        #assert len(true_set[i])==len(pred_set[i])
        
    print(no)
    
    y_true = []
    for i in range(len(true_set)):
        if i not in no:
            y_true += true_set[i]
    y_pred = []
    for i in range(len(pred_set)):
        if i not in no:
            y_pred += pred_set[i]
            
    print(pred_file)        
        
    p,r,f = f1_partial(y_true,y_pred,weight = 1)
    print('p:',p,' r:',r,' f:',f)
        
    p,r,f = f1_partial(y_true,y_pred)
    print('p:',p,' r:',r,' f:',f)
    
    p,r,f = f1_partial(y_true,y_pred,weight = 0)
    print('p:',p,' r:',r,' f:',f)
    
    p,r,f = f1_intersection(y_true,y_pred)
    print('p:',p,' r:',r,' f:',f)
    

     


