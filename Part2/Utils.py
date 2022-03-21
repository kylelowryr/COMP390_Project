from PIL import Image
import os


def create_image(file_path):
    image = Image.new('RGB', (224, 224), (255, 255, 255))
    image.save(file_path)
    image.show()


def amend_off():
    test_num1 = 201
    test_num2 = 286
    train_num1 = 1
    train_num2 = 200
    for i in range(1, 889):
        strnum = str(i).zfill(4)
        name = 'train\\chair_' + strnum + '.off'
        list1 = []
        with open(name, 'r+') as f:
            lines = f.readlines()
            length = len(lines)
            print(length)
            line_1 = lines[0]
            print(line_1)
            if line_1 != 'OFF\n':
                print('not OFF')
                line_1_num = line_1.split('OFF')[-1]
                line_1_final = line_1_num.split('\n')[0]
                # print(line_1)
                list1.append('OFF')
                list1.append(line_1_final)
                print(line_1_final)
                for i in range(1, length):
                    str1 = lines[i].split('\n')[0]
                    list1.append(str1)
                print(len(list1))
        f.close()
        if line_1 != 'OFF\n':
            with open(name, 'w') as f:

                for i in range(len(list1)):
                    f.write(list1[i] + '\n')
            f.close()


def write2set():
    with open('testset.txt','w') as f :
        for i in range(156,256):
            i_1 = str(i)
            num = i_1.zfill(4)

            f.write(".\ModelNet40\\guitar\\test\\guitar_"+num+".off\n")
        f.close()

def initialization():
    classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
    class_args_dic = {'airplane':(1,626,627,726),
             'bathtub':(1,106,107,156),
             'bed':(1,515,516,615),
             'bench': (1, 173,174,193),
             'bookshelf':(1,572,573,672),
             'bottle': (1,335,336,435),
             'bowl': (1,64,65,84),
             'car': (1,197,198,297),
             'chair': (1,889,890,989),
             'cone': (1,167,168,187),
             'cup': (1,79,80,99),
             'curtain': (1,138,139,158),
             'desk': (1,200,201,286),
             'door': (1,109,110,129),
             'dresser': (1,200,201,286),
             'flower_pot': (1,149,150,169),
             'glass_box': (1,171,172,271),
             'guitar': (1,155,156,255),
             'keyboard': (1,145,146,165),
             'lamp': (1,124,125,144),
             'laptop': (1,149,150,169),
             'mantel': (1,284,285,384),
             'monitor': (1,465,466,565),
             'night_stand': (1,200,201,286),
             'person': (1,88,89,108),
             'piano': (1,231,232,331),
             'plant': (1,240,241,340),
             'radio': (1,104,105,124),
             'range_hood': (1,115,116,215),
             'sink': (1,128,129,148),
             'sofa': (1,680,681,780),
             'stairs': (1,124,125,144),
             'stool': (1,90,91,110),
             'table': (1,392,393,492),
             'tent': (1,163,164,183),
             'toilet': (1,344,345,444),
             'tv_stand': (1,267,268,367),
             'vase': (1,475,476,575),
             'wardrobe': (1,87,88,107),
             'xbox': (1,103,104,123),
             }
    key_list = []
    for key in class_args_dic:
        key_list.append(key)
    print(key_list)
    return class_args_dic,key_list

def gen_txt():
    dic,key_list = initialization()
    with open('testset.txt', 'w') as f:
        for i in range(len(key_list)):
            train_start=dic[key_list[i]][0]
            train_end = dic[key_list[i]][1]+1
            test_start = dic[key_list[i]][2]
            test_end = dic[key_list[i]][3] + 1
            for id1 in range(train_start,train_end):
                id1_1 = str(id1)
                num_id1 = id1_1.zfill(4)
                f.write(".\ModelNet40\\"+str(key_list[i]) +"\\train\\"+str(key_list[i]) +'_'+ num_id1 + ".off\n")
            for id2 in range(test_start,test_end):
                id2_1 = str(id2)
                num_id2 = id2_1.zfill(4)
                f.write(".\ModelNet40\\"+str(key_list[i]) +"\\test\\"+str(key_list[i]) + '_'+num_id2 + ".off\n")
        f.close()

