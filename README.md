# README

# 1. Set-up

---

These basic settings are used in my local implementation.

Environment:

*Python - 3.8.8*

*Pytorch - v1.10.1*

# 2. Start and initialization

---

## 2.1 Preparing models MVCNN/View-GCN

These two models can be accessed in github. See: [https://github.com/suhangpro/mvcnn](https://github.com/suhangpro/mvcnn) and [https://github.com/weixmath/view-GCN](https://github.com/weixmath/view-GCN). Basing on their README file to check the availability and whether it is executable.

## 2.2 Preparing datasets

Whether use provided datasets or generate by yourself. Substitute the original files in the previous 2 model projects.

### 2.2.1 Generate by yourself

Using blender, blenderphong, off-addon in combination to generate ModelNet 40 in your ways.

## 2.3 Training models

Call training programs to train the model in your configs.

### 2.3.1 Using pre-trained .pth file.

I provide the collection of .pth file in different configs. 

Using codes to load.

<model_net>.load_state_dict(torch.load('xxxxxxxx.pth'))

## 2.4 Add Part-1 to the model project

Part1 contains a util class and Predictor class.

Creating a predictor:

predictor = ModelNetPredictor(model=cnet_2, pred_list=generate_dataset(20, args), model_name='view-gcn', num_views=20)

Using merge to directly loop the dataset, using different covers.

predictor.merge()

It will recorded in a .csv file.

### 2.4.1 New Argument - cover

Cover means ‘cover a view’, to make a view all_white/all_black depends on your choice( the create_image function is in the Part2)

cover = -1 means no cover, from 0 - 19 means the specific view you want to cover.

### 2.4.2 New Argument - start

In the beginning, I try to find whether the start position will influence the output, so this argument left, the ‘start’ means the first image in a group to input. Changing the sequence.

### 2.4.3 New Argument - disturb

disturb= 0 means no disturbance, disturb = n >0 means in the mid (112,112) area, randomly select n pixels to be normalized as a white pixel.

## 2.5 Part-2 Analysing

Part2 looks like a separated part,  can be regarded as a new project.

### 2.5.1 Load_data/Load_opt()

load whole .csv data or load 1 specific class from whole data.

### 2.5.2 Normalize()

1st Normalization func, try to make data which in one row to [-1,1] and also make it reversed.

### 2.5.2 Normalize2()

2nd Normalization, make data in a class to [-1,1]. The output looks like ‘Weights’

### 2.5.3 test_visual()

Basic visualization func

### 2.5.4 test1l()/analyse_noWeight()

Analysing one row/one class of data, the early thoughts.

### 2.5.5 Utils - Analysis, manipulate

Output max, min and multiplication.

### 2.5.6analyse_withWeight()/analyse_withWeight_all()/analyse_withWeight_Group()

These funcs are basing on Weights, the main idea now. Calculating the local max,min and make normalization to get weights. Through row normalization and weights to get the score of each view/view-group.

### 2.5.7 Other Utils - Helpful for generating datasets.

Standardization of .off data, generating .txt file for cmd command and so on.

# 3. Webpages

---

[https://kylelowryr.github.io/](https://kylelowryr.github.io/)

# 4. Sources

---

### Models

MVCNN: [https://github.com/suhangpro/mvcnn](https://github.com/suhangpro/mvcnn)

View-GCN: ‣[https://github.com/weixmath/view-GCN](https://github.com/weixmath/view-GCN)

### Datasets

ModelNet: [https://modelnet.cs.princeton.edu/](https://modelnet.cs.princeton.edu/)

### Tools

Blender tools: 

[https://github.com/zeaggler/ModelNet_Blender_OFF2Multiview](https://github.com/zeaggler/ModelNet_Blender_OFF2Multiview)

[https://github.com/alextsui05/blender-off-addon](https://github.com/alextsui05/blender-off-addon)

### Personal/Prepared sources

Datasets:

ModelNet40 - 20views - shade: [https://drive.google.com/file/d/1EK7ApY3f_LAy8x1GlFfFFDVnpg8wJIaS/view?usp=sharing](https://drive.google.com/file/d/1EK7ApY3f_LAy8x1GlFfFFDVnpg8wJIaS/view?usp=sharing)

ModelNet40 - 20views - Noshade: [https://drive.google.com/file/d/1AO_RQGQ3_aoXpbqzdpGXUC6x4tafuuZy/view?usp=sharing](https://drive.google.com/file/d/1AO_RQGQ3_aoXpbqzdpGXUC6x4tafuuZy/view?usp=sharing)

Pretrained .pth files collection with different configs: [https://drive.google.com/file/d/1ejeF8C7n47Chzkt-6iw4NJtkntbWAnyc/view?usp=sharing](https://drive.google.com/file/d/1ejeF8C7n47Chzkt-6iw4NJtkntbWAnyc/view?usp=sharing)

Output .csv files collection: [https://drive.google.com/file/d/1iRep8bsAQ0ze1BvbPyoii5JKi81qYK_U/view?usp=sharing](https://drive.google.com/file/d/1iRep8bsAQ0ze1BvbPyoii5JKi81qYK_U/view?usp=sharing)
