#%%
import pickle
import numpy as np
import os
import jieba

class pred_clas():
    '''
    path = pickle存取位置

    '''
    #add notes / path object!!! (get_cat)
    def __init__(self, path = os.getcwd() + "/job1111_pred.pkl"):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def dummy_f(doc):
        return doc

    def tokenize(text):
        text = text.strip()
        tokens = jieba.lcut(text, cut_all=False, HMM=True)
        tokens = ' '.join(tokens).split()
        return tokens

    def get_category(self, x):
        model = self.model
        if type(x) is list:
            new_list = []
            for i in list:
                i = tokenize(i)
                new_list.append(i)
            x = new_list   
        else:
            x = tokenize(x)
            x=[x]
        cat = model.predict(x)
        cat = np.array2string(cat)
        cat = cat[2:-2]
        return cat


#%%
classifier = pred_clas()
ex1 = "會計"
cat = classifier.get_category(ex1)
print("職稱: {} \n分類: {}".format(ex1, cat))

# %%
