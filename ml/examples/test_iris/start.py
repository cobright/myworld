#%%

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mglearn

#%%
db = load_iris()
X_train, X_test, y_train, y_test = train_test_split(db['data'],db['target'], random_state=0)
#데이터 살펴보기
#df = pd.DataFrame(X_train, columns=db['feature_names'])
#pd.plotting.scatter_matrix(df, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8)

#%%
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

#%%
X_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(X_new)
print("예측: {}".format(prediction))
print("타깃: {}".format(db['target_names'][prediction]))

#%%
y_pred = knn.predict(X_test)
print("test set:{}".format(y_pred))

#%%

print("test set accuracy:{:.2f}".format(knn.score(X_test, y_test)))

#%%


#%%
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0], X[:,1], y)

#%%
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylabel("Target")
plt.xlabel("Feature")

#%%
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print("sample count per class:\n{}".format(
    {n:v for n, v in zip(cancer.target_names,np.bincount(cancer.target))}
))

#%%
print("feature name\n{}".format(cancer.feature_names))

#%%
cancer.DESCR

#%%
from sklearn.datasets import load_boston
boston = load_boston()
boston.data.shape
boston.DESCR

#%%
X,y=mglearn.datasets.load_extended_boston()
X.shape

#%%
mglearn.plots.plot_knn_classification(n_neighbors=3)

#%%
X,y=mglearn.datasets.make_forge()
X_train,X_test, y_train, y_test = train_test_split(X,y,random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
#%%
knn.predict(X_test)
#%%
knn.score(X_test, y_test)
#%%
fig, axes = plt.subplots(1,3,figsize=(10,3))

for n_neighbors, ax in zip([1,3,9],axes):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(knn,X, fill=True, eps=0.5, ax=ax,alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title(n_neighbors)

#%%
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train,X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,
stratify=cancer.target,random_state=0)
training_accuracy =[]
test_accuracy =[]
neighbors_settings = range(1,11)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)    
    training_accuracy.append(knn.score(X_train,y_train))
    test_accuracy.append(knn.score(X_test,y_test))

plt.plot(neighbors_settings,training_accuracy, label="Training Accuracy")
plt.plot(neighbors_settings,test_accuracy,label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
