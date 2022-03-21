import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import heapq


def load_data(file_path):
    df = pd.DataFrame(pd.read_csv(file_path,index_col=0))
    #print(df)
    return df


def df_opt(df,num_class_spec):
    model = [100,50,100,20,100,100,20,100,100,20,20,20,86,20,86,20,100,100,20,20,20,100,100,86,20,100,100,20,100,20,100,20,20,100,20,100,100,100,20,20]
    model_index =[]
    temp_sum=0
    model_index.append(temp_sum)
    for i in range(len(model)):
        temp_sum = temp_sum + model[i]
        model_index.append(temp_sum)
    df_new = df.iloc[model_index[num_class_spec]:model_index[num_class_spec+1]]

    return df_new


def normalize(pred_list):
    pos=[]
    neg=[]
    pos_neg =[]
    norm_pred=[]
    max_pos,min_pos,max_neg,min_neg = 0,0,0,0
    for _,data in enumerate(pred_list.copy()):
        if data >0:
            index = 'pos'
            pos.append(data)
        else:
            index = 'neg'
            neg.append(data)
        pos_neg.append(index)
    if len(pos) == 0:
        pass
    else:
        max_pos = max(pos)
        min_pos = 0
        diff_pos = max_pos-min_pos
    if len(neg) == 0:
        pass
    else:
        max_neg = 0
        min_neg = min(neg)
        diff_neg = max_neg - min_neg
    for i in range(len(pos_neg)):
        if pos_neg[i] == 'pos':
            if diff_pos == 0:
                temp = 1
            else:
                temp  = (pred_list[i]-min_pos)/diff_pos
        else:
            if diff_neg == 0:
                temp = -1
            else:
                temp = (pred_list[i]-max_neg)/diff_neg
        temp = round(temp,4)
        norm_pred.append(temp)
    norm_pred_reversed=[i * (-1) for i in norm_pred]
    return norm_pred_reversed


def normalize2(pred_list,max_diff,min_diff):
    pos = []
    neg = []
    pos_neg = []
    norm_pred = []
    max_pos, min_pos, max_neg, min_neg = 0, 0, 0, 0
    for _, data in enumerate(pred_list.copy()):
        if data >= 0:
            index = 'pos'
            pos.append(data)
        else:
            index = 'neg'
            neg.append(data)
        pos_neg.append(index)
    if len(pos) == 0:
        pass
    else:
        max_pos = max_diff
        min_pos = 0
        diff_pos = max_pos - min_pos
    if len(neg) == 0:
        pass
    else:
        max_neg = 0
        min_neg = min_diff
        diff_neg = max_neg - min_neg
    for i in range(len(pos_neg)):
        if pos_neg[i] == 'pos':
            if diff_pos == 0:
                temp = 1
            else:
                temp = (pred_list[i] - min_pos) / diff_pos
        else:
            if diff_neg == 0:
                temp = -1
            else:
                temp = (max_neg-pred_list[i]) / diff_neg
        temp = round(temp, 4)
        norm_pred.append(temp)
    norm_pred_1 = [i  for i in norm_pred]
    return norm_pred_1


def test_visual(pred_list,test_num,num_views,threshold,log_dir):
    plt.figure('1')
    plt.title('Scatter graph of No. %d ' % test_num)
    x=[]
    y=pred_list
    #print(y)
    ax = plt.gca()
    for i in range(num_views):
        x.append(i)
        x_ = i
        y_ = pred_list[i]
        #print('i %d , pred %.4f'% (x_,y_))
    plt.xticks(np.arange(0,20,1))
    plt.yticks(np.arange(-1, 1.05, 0.2))
    #print(x)
    #print(y)
    plt.plot(x,y,color = 'grey')
    a=[]
    for i in range(num_views):
        if y[i] >=threshold:
            plt.plot(x[i], y[i], color='red', marker='D', markersize=8, markeredgecolor='red')
            a.append(i+1)
        else:
            plt.plot(x[i], y[i], color='grey', marker='.', markersize=10, markeredgecolor='grey')
    plt.plot(x, np.zeros(num_views))
    plt.plot(x, np.ones(num_views)*threshold)
    #print('a:',a)
    plt.savefig(log_dir)
    plt.show()


def test1(df,row):
    Length_cols = df.shape[1]
    diff_target_list = []
    diff_pred_list = []
    bool_ = np.zeros(20)
    wrong_ =[]
    wrong_cover = []
    list_test =[]
    for i in range(int(Length_cols/4)):
        # print(i)
        mode = 0
        predValue = df.iloc[row][4 * i]
        correctValue = df.iloc[row][4 * i + 1]
        targetValue = df.iloc[row][4 * i + 2]
        combLabel = df.iloc[row][4 * i + 3]
        str1 = str(combLabel)
        # print(str1)
        str_p1 = str1.split(',')[0]
        str_p2 = str1.split(',')[-1]
        # print(str_p1,'   ',str_p2)
        if correctValue == False:
            wrong_.append(str_p1)
            wrong_cover.append(i)
            list_test.append((i,str_p1))

        if i == 0:
            basePredValue = df.iloc[row][4 * i]
            baseCorrectValue = df.iloc[row][4 * i + 1]
            baseTargetValue = df.iloc[row][4 * i + 2]
            baseCombLabel = df.iloc[row][4 * i + 3]
            baseLabel=str_p2
            cover = 'all'

        else:
            cover = str(i)
            if baseCorrectValue != correctValue:
                bool_variation = True
                bool_[i-1]=False
            else:
                bool_variation = False
                bool_[i-1] = True
            diff_target = targetValue-baseTargetValue
            diff_pred = predValue - basePredValue
            diff_target_list.append(diff_target)
            diff_pred_list.append(diff_pred)
    if len(set(bool_)) == 1:
        mode = 1
    else:
        mode = 2
    # print(wrong_)
    # print(wrong_cover)
    # print(baseLabel)
    # print(diff_pred_list)
    # print(diff_target_list)
    # print(bool_.tolist())
    # print(mode)
    # print(list_test)
    l = list([baseLabel,bool_.tolist(),list_test])
    # for var in l:
    #     print(var)
    return baseLabel,diff_pred_list,diff_target_list


def analyse_noWeight(df,threshold):
    Length_rows = df.shape[0]
    count = np.zeros(20)
    for i in range(Length_rows):
        label, diff1, diff2 = test1(df, row=i)
        diff1_norm1 = normalize(diff1)
        diff2_norm1 = normalize(diff2)
        for n in range(len(diff2_norm1)):
            if diff2_norm1[n] >= threshold:
                count[n] = count[n]+1

    count_new = [(x/sum(count)) for x in count]
    print(count_new)


def analyse_withWeight(df,max_all,min_all):
    Length_rows = df.shape[0]
    count = np.zeros(20)
    num = np.zeros(20)
    num_group = []
    for i in range(Length_rows):
        label, diff1, diff2 = test1(df, row=i)
        diff1_norm1 = normalize(diff1)
        diff2_norm1 = normalize(diff2)
        diff1_norm2 = normalize2(diff2,max_diff=max_all,min_diff=min_all)
        diff2_norm2 = normalize2(diff2,max_diff=max_all,min_diff=min_all)
        mani_ = manipulate(diff2_norm1,diff2_norm2)
        for n in range(len(mani_)):
            num[n] = num[n] + mani_[n]
    #print(num)
    for i in range(int(len(num)/5)):
        num_group.append(sum(num[i:i+4]))
    num_norm = normalize(num)
    num_norm_new = [i*(-1) for i in num_norm]
    num_group_new = [i / (5) for i in num_group]
    num_group_new_norm = normalize(num_group_new)
    num_group_new_norm_new = [i*(-1) for i in num_group_new_norm]
    index_1 = num_group_new_norm_new.index(max(num_group_new_norm_new))
    return num_norm_new,num_group_new_norm_new,index_1

def analysis(df):
    Length_rows = df.shape[0]
    count = np.zeros(20)

    for i in range(Length_rows):
        label, diff1, diff2 = test1(df, row=i)

        #print(diff2_norm)
        if i == 0:
            min_ = min(diff2)
            max_ = max(diff2)
        else:
            temp_min = min(diff2)
            temp_max = max(diff2)
            if min_ >= temp_min:
                min_ = temp_min
                # print(i)
            if temp_max >= max_:
                max_ = temp_max
            # print(i)
    # print(min_)
    # print(max_)
    return max_,min_


def manipulate(x,y):
    z=[]
    for i in range(len(x)):
        z_ = x[i]*y[i]
        z.append(z_)
    return z


def Summarize(norm,num_class_spec,threshold_list):
    results_all =[]

    print('No.',num_class_spec)
    for i in range(len(threshold_list)):
        local_th = threshold_list[i]
        results = []
        for n in range(len(norm)):
            if norm[n] >= local_th:
                results.append(n+1)
        results_all.append(results)
    for j in range(len(results_all)):
        print(results_all[j])
    print('\n')
    return results_all


def analyse_withWeight_all(df,th_list,num_class,top_k,num_view):
    count1 = np.zeros(20)
    count2 = np.zeros(20)
    print('Start summarizing over classes, total classes: %d ' % (num_class))
    for i in range(0, num_class):
        df_test = df_opt(df, num_class_spec=i)
        max1, min1 = analysis(df_test)
        num_norm, num_group, index = analyse_withWeight(df_test, max_all=max1, min_all=min1)
        results = Summarize(norm=num_norm, num_class_spec=i,threshold_list=th_list)
        for p in range(len(results[0])):
            count1[results[0][p] - 1] = count1[results[0][p] - 1] + 1
        for p1 in range(len(results[1])):
            count2[results[1][p1] - 1] = count2[results[1][p1] - 1] + 1
    print('Start summarizing over views, total views: %d'% (num_view))
    for q in range(len(count1)):
        print('View:', q + 1)
        print('counts for threshold > 0.75: ',count1[q])
        print('counts for threshold > 0.5: ',count2[q])

    #print(count1)
    #print(count2)
    top_k_idx1 = count1.argsort()[::-1][0:top_k]
    top_k_idx2 = count2.argsort()[::-1][0:top_k]
    #print(top_k_idx1)
    #print(top_k_idx2)
    max_number1 = heapq.nlargest(top_k, count1)
    max_number2 = heapq.nlargest(top_k, count2)
    max_index1 = []
    max_index2 = []
    c1list1 = count1.tolist()
    c1list2 = count2.tolist()
    for t1 in max_number1:
        index1 = c1list1.index(t1)
        max_index1.append(index1)
        c1list1[index1] = 0
    for t2 in max_number2:
        index2 = c1list2.index(t2)
        max_index2.append(index2)
        c1list2[index1] = 0
    print('Max 1 - threshold > 0.75: ')
    print(max_number1)
    print(max_index1)
    print('Max 1 - threshold > 0.5: ')
    print(max_number2)
    print(max_index2)


def analyse_withWeight_Group(df,num_class):
    test=[]
    for i in range(0,num_class):
        df_test = df_opt(df,num_class_spec=i)
        max1, min1 = analysis(df_test)
        num_norm, num_group,index = analyse_withWeight(df_test, max_all=max1, min_all=min1)
        #print('index',i,'',index)
        test.append(index)
    se = pd.Series(test)
    countDict = dict(se.value_counts())
    print(countDict)


if __name__ == '__main__':
    df = load_data(file_path='archive_View-GCN/views_ep10_shade_white_withlabel.csv')
    num_class_test = 0
    df_test1= df_opt(df,num_class_spec= num_class_test )
    #label,diff1,diff2 = test1(df,row=0)

    #test_visual(pred_list=x, test_num=row_test, num_views=20, threshold=0.75)
    #test_visual(pred_list=manipulate(x,x1),test_num=0,num_views=20,threshold=0.75)
    # analyse_noWeight(df, threshold=1)
    # analyse_noWeight(df, threshold=0.9)
    # analyse_noWeight(df, threshold=0.8)
    # analyse_noWeight(df, threshold=0.7)
    # analyse_noWeight(df, threshold=0.6)
    # analyse_noWeight(df, threshold=0.5)
    max_test1,min_test1 = analysis(df_test1)
    analyse_withWeight_Group(df,num_class=40)
    num_norm,num_group,index = analyse_withWeight(df_test1,max_all=max_test1,min_all=min_test1)
    analyse_withWeight_all(df,th_list=[0.75,0.5,0],num_class=40,top_k=10,num_view=20)

    test_visual(pred_list=num_norm, test_num=num_class_test, num_views=20, threshold=0.75,log_dir='th0.75.jpg')
    test_visual(pred_list=num_norm, test_num=num_class_test, num_views=20, threshold=0.5,log_dir='th0.5.jpg')
    test_visual(pred_list=num_norm, test_num=num_class_test, num_views=20, threshold=0,log_dir='th0.jpg')

