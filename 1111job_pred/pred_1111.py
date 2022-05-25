#%%
import pickle
import numpy as np
import os

def dummy_f(doc):
    return doc

class pred_clas():

    def __init__(self, path = os.getcwd() + "/job1111_pred.pkl"):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def get_category(self, x):
        model = self.model
        if type(x) is list:
            pass
        else:
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
