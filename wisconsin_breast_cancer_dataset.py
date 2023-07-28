import mglearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn.datasets

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
#print("Cancer keys: \n{}".format(cancer.keys()))
#print("Shape of cancer data: {}".format(cancer.data.shape))    # (x,y) indicates x samples and y features     
#print("Sample counts per class;\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
#print("Feature names: {}".format(cancer.feature_names))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(cancer['data'], cancer['target'], random_state=0)

cancer_dataframe = pd.DataFrame(X_train, columns= cancer.feature_names)

cancer_combined = pd.concat([cancer_dataframe, pd.DataFrame(y_train, columns=['species'])], axis=1)

scm = sns.pairplot(cancer_combined, hue='species')
#plt.show()
max_score = 0.90
from sklearn.neighbors import KNeighborsClassifier
for k in range(1, 50):
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(X_train, y_train)
    y_pred = kn.predict(X_test)
    
    #print("Test set predictions:\n {}".format(y_pred))
    #print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
    #print("\n")
    #print(k)
    #print("\n")
    if(np.mean(y_pred == y_test)>=max_score):
        max_score = np.mean(y_pred == y_test)
        final_k = k
    
    #print("\n")
print("Max. accuracy is: {}".format(max_score))
print("  at value of n_neighbors: {}" .format(final_k))



