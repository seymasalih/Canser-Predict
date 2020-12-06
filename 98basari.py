

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
veri= pd.read_csv("breast-cancer-wisconsin.data")
veri.replace("?", -99999 ,inplace=True)#? leri -9999 ile değiştirdik
veri.drop(['id'], axis=1)#id sütununu gereksiz olduğundan çıkardık
y= veri.benormal
x=veri.drop(['benormal'],  axis=1)
imp= SimpleImputer(missing_values=-99999, strategy="mean")

X_train, X_test, y_train, y_test= train_test_split(x,y, test_size=0.2)
x = imp.fit_transform(x)
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

"""for z in range(25):
    z = 2*z+1
    print("En yakın",z,"komşu kullandığımızda tutarlılık oranımız")
    tahmin = KNeighborsClassifier(n_neighbors=z, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1)
    tahmin.fit(x,y)
    ytahmin = tahmin.predict(x)

    basari = accuracy_score(y, ytahmin, normalize=True, sample_weight=None)
    print(basari)"""
    
tahmin= KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1)
tahmin.fit(X_train , y_train)
ytahmin=tahmin.predict(x)
basari= tahmin.score(X_test, y_test)
print("Yüzde",basari*100," oranında:" )
a= np.array([1,2,2,2,3,2,1,2,3,2]).reshape(1,-1)
print(tahmin.predict(a))
