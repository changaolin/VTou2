import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from io import open
import numpy as np
sep=' '
flags = r'[。！？]'
import re
line_max = 20
max_len = 60
def word2tag():
    with open("BosonNLP_NER_6C.txt",'r',encoding='utf-8') as input_data ,open("tmp.txt","w",encoding='utf-8') as output_data:
        for line in input_data.readlines():
            line = line.strip()
            i = 0
            while i < len(line):
                if line[i] == '{':
                    i+=2
                    temp = ""
                    while line[i]!="}":
                        temp+=line[i]
                        i+=1
                    i+=2
                    word=temp.split(":")
                    sen=word[1]
                    output_data.write(sen[0]+"/B_"+word[0]+sep)
                    for j in sen[1:len(sen)-1]:
                        output_data.write(j+"/M_"+word[0]+sep)
                    output_data.write(sen[-1]+"/E_"+word[0]+sep)
                else:
                    output_data.write(line[i]+"/O"+sep)
                    i+=1

def splitText():
    with open("tmp.txt",'r',encoding='utf-8') as readin,\
        open("result.txt",'w',encoding='utf-8') as writeto:
        for line in readin.readlines():
            if len(line.strip()) == 0:
                continue
            lines = re.split(flags,line)
            for l in lines:
                if len(l) > line_max:
                    ll = l.split("，")
                    writeto.write("\n".join(ll))
                else:
                    writeto.write("\n"+l)
        pass

def savePkl():
    datas = []
    labels = []
    tags = set()
    with open('result.txt','r',encoding='utf-8') as inp:
        for line in inp.readlines():
            line = line.split()
            linedata = []
            linelabel = []
            numNotO = 0
            for word in line:
                word = word.split('/')
                linedata.append(word[0])
                linelabel.append(word[1])
                tags.add(word[1])
                if word[1] != 'O':
                    numNotO += 1
            if numNotO != 0:
                datas.append(linedata)
                labels.append(linelabel)
    print(len(datas),tags)
    print(len(labels))
    new_datas = []
    for data in datas:
        new_datas.extend(data)
    allwords = pd.Series(new_datas).value_counts()
    set_words = allwords.index

    set_ids = range(1,len(set_words)+1)

    tags = [i for i in tags]
    tag_ids = range(len(tags))
    word2id = pd.Series(set_ids,index=set_words)
    id2word = pd.Series(set_words,index=set_ids)
    tag2id = pd.Series(tag_ids,index=tags)
    id2tag = pd.Series(tags,index=tag_ids)

    word2id["unknow"] = len(word2id)+1
    def padding(type,words):
        if type == "X":
            ids = list(word2id[words])
        if type == "Y":
            ids = list(tag2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0]*(max_len - len(ids)))
        return ids

    df_datas = pd.DataFrame({"words":datas,"tags":labels},index=list(range(len(datas))))
    df_datas['x'] = df_datas["words"].apply(lambda x:padding("X", x))
    df_datas['y'] = df_datas["tags"].apply(lambda x:padding("Y", x))
    x = np.asarray(list(df_datas['x'].values))
    y = np.asarray(list(df_datas['y'].values))
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=12)
    def dumpPkl(ll,out):
        for l in ll:
            pickle.dump(l,out)
    with open("../BosonNLPtmp.pkl",'wb') as out:
        x_train = x
        y_train = y
        ll = [word2id,id2word,tag2id,id2tag,x_train,y_train,x_test,y_test]
        dumpPkl(ll,out)
        pass





if __name__ == '__main__':
    word2tag()
    splitText()
    savePkl()