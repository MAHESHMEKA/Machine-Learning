import numpy as np

class KNeighborsClassifier:

    def __init__(self,n_neighbors=3):
        self.n_neighbors=n_neighbors

    def fit(self,x_train,y_train):
        self.x_train=np.array(x_train,dtype=float)
        self.y_train=y_train

    def predict(self,x_samples,metric="euclidean",p=1):
        predictions=[]
        for x_sample in x_samples:
            if metric.lower()=="euclidean":
                p=2
            if metric.lower()=="manhattan":
                p=1
            distances=(np.sum(abs((self.x_train-x_sample))**p,axis=1))**(1/p)
            k_nearest=np.argsort(distances)[:self.n_neighbors]
            labels=self.y_train[k_nearest] 
            unique_labels,count=np.unique(labels,return_counts=True)
            key=unique_labels[list(count).index(max(count))].item() 
            predictions.append(key)
        return predictions
