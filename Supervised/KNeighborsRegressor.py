import numpy as np

class KNeighborsRegressor:

    def __init__(self,n_neighbors=3):
        self.n_neighbors=n_neighbors

    def fit(self,x_train,y_train):
        self.x_train=np.array(x_train,dtype=float)
        self.y_train=np.array(y_train,dtype=float)

    def predict(self,x_samples,metric="euclidean",p=1):
        predictions=[]
        for x_sample in x_samples:
            if metric.lower()=="euclidean":
                p=2
            if metric.lower()=="manhattan":
                p=1
            distances=(np.sum(abs((self.x_train-x_sample))**p,axis=1))**(1/p)
            k_nearest=np.argsort(distances)[:self.n_neighbors]
            values=self.y_train[k_nearest] 
            predictions.append(np.mean(values).item())
        return predictions
