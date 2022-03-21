import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
import time
import pandas as pd


class ModelNetPredictor(object):

    def __init__(self, model, pred_list,model_name,num_views=20):
        self.model = model
        self.pred_list = pred_list
        self.model_name = model_name
        self.num_views = num_views
        self.model.cuda()


    def pred(self,cover):
        out_data = None
        in_data = None
        TF_list = []


        self.model.eval()
        name_cols = self.initializa(cover)
        LENGTH = len(self.pred_loader.dataset)
        print("*"*30)
        print(LENGTH)
       # name_cols = self.initializa(cover)
        df_all = pd.DataFrame(columns=name_cols)

        for i, data in enumerate(self.pred_loader):

            N, V, C, H, W = data[1].size()
            in_data = Variable(data[1]).view(-1, C, H, W).cuda()
            target = Variable(data[0]).cuda().cpu()
            target_index = Variable(data[0]).cuda().cpu().numpy()[0]
            if self.model_name == 'mvcnn':
                out_data = self.model(in_data)
            else:
                out_data,_,_ = self.model(in_data)
            #print(out_data.size())
            #print(out_data)
            pre_test = torch.max(out_data, 1)
            #print(pre_test)
            pred_softmax = torch.softmax(out_data, 1)

            pred_softmax_sorted=torch.sort(pred_softmax, descending=True)[1][0].cpu().detach().numpy()
            #print(pred_softmax_sorted)
            position_index = np.argwhere(pred_softmax_sorted == target_index)[0,0]
            #print(position_index)
            #print('test')
            #print(torch.max(pred_softmax, 1)[0])
            pred_first_value = torch.max(pred_softmax, 1)[0].cpu().detach().numpy()[0]
            pred_first_index = torch.max(pred_softmax, 1)[1].cpu().detach().numpy()[0]
            pred_true_value = pred_softmax[0][target_index].cpu().detach().numpy()
            bool_label = target_index == pred_first_index
            if i <= 10:

                #print('---> Summary Session Starts:')
                #print('No. %d ' % i)
                #print("target label is: ", target_index)
                #print('Label position: %d / 40 ' % (position_index + 1))
                #print('Label correctness: ',bool_label)
                #print('Prob with # 1-High, Label: %d, prob_value: %.4f ' % (pred_first_index,pred_first_value*100))
                #print('Prob with # True  , Label: %d, prob_value: %.4f \n' % (target_index,pred_true_value * 100))
                #print('Model No. %d , target is %d , T/F is %s , pred_value is %.4f ' % (i,target_index,bool_label,pred_true_value * 100) )
                a = [[i,target_index,pred_first_index,bool_label,pred_true_value * 100]]
                top5_prob = torch.topk(pred_softmax, 5, 1)[0]

            Label = str(pred_first_index)+','+str(target_index)
            df_all.loc[i] = {name_cols[0]: pred_first_value * 100, name_cols[1]: bool_label, name_cols[2]: pred_true_value*100,name_cols[3]:Label }

            #print(df_all)
            #self.df.loc[i]={'Model-num':i,'Class-label':target_index,'Pred-val-12':pred_true_value * 100}

        #print(self.df)
            #print(top5_prob)
            #print("---> End Session\n")
        return df_all


    def merge(self):
        start1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(start1)
        df_pred_all = self.pred(cover=-1)
        df_list = []

        df_list.append(df_pred_all)
        for i in range(0,self.num_views):
            print(i)

            start = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print(start)
            df_pred_name = 'df_pred_c'+str(i)
            locals()[df_pred_name] = self.pred(cover=i)
            df_list.append(locals()[df_pred_name])
        df_f = pd.concat(df_list,axis=1)
        print(df_f)
        df_f.to_csv('views_ep1_noshade_black_withlabel.csv')


    def initializa(self,cover):
        if cover == -1:
            self.pred_loader = self.pred_list[0]
            cols = ['Pred-all','T/F-all','Pred-True-all','Label-all']
            #df_all = pd.DataFrame(columns=['Pred-all','T/F-all'])
        else:
            for i in range(0, self.num_views):
                if cover == i:
                    str1 = 'Pred-c'+str(i)
                    str2 = 'T/F-c' + str(i)
                    str3 = 'Pred-True-c' + str(i)
                    str4 = 'Label-c'+str(i)
                    cols = [str1,str2,str3,str4]
                    self.pred_loader = self.pred_list[i]
        return cols