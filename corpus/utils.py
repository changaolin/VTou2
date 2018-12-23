from io import open
import numpy as np
import re
sep=' '
max_len = 60
flags = '[，。！？；]'
def get_entity(x,y,id2tag):
    entity=""
    res=[]
    for i in range(len(x)): #for every sen
        for j in range(len(x[0])): #for every word
            if y[i][j]==0:
                continue
            if id2tag[y[i][j]][0]=='B':
                entity=id2tag[y[i][j]][2:]+':'+x[i][j]
            elif id2tag[y[i][j]][0]=='M' and len(entity)!=0 :
                entity+=x[i][j]
            elif id2tag[y[i][j]][0]=='E' and len(entity)!=0 :
                entity+=x[i][j]
                res.append(entity)
                entity=[]
            else:
                entity=[]
    return res
def padding(ids):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        ids.extend([0]*(max_len-len(ids)))
        return ids

def padding_word(sen):
    if len(sen) >= max_len:
        return sen[:max_len]
    else:
        return sen

def test_input(model,sess,word2id,id2tag,batch_size):
    while True:
        text = input("Enter your input: ")
        # flag = u'；。！？"\\n"'
        text = re.split(flags, text)
        text_id=[]
        for sen in text:
            word_id=[]
            for word in sen:
                if word in word2id:
                    word_id.append(word2id[word])
                else:
                    word_id.append(word2id["unknow"])
            text_id.append(padding(word_id))
        zero_padding=[]
        zero_padding.extend([0]*max_len)
        text_id.extend([zero_padding]*(batch_size-len(text_id)))
        feed_dict = {model.input_data:text_id}
        pre = sess.run([model.viterbi_sequence], feed_dict)
        entity = get_entity(text,pre[0],id2tag)
        print( 'result:')
        for i in entity:
            print (i)
    pass