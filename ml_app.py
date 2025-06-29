import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.write("# Explore different ml models and their accuracy on different dataset")

dataset_name = st.sidebar.selectbox(
    'Select dataset',
    ['Iris', 'Wine', 'Breast Cancer']
)

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ['Knn', 'SVM', 'Random Forest']
)

# defining a funtion to load different datasets 
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y

# callin the function to load the dataset
X,y = get_dataset(dataset_name)

# to print the shape of the dataset
st.write('Shape of the dataset:', X.shape)
st.write('Number of classes:', len(np.unique(y)))

# defining different classifiers parameters
def add_paramater_ui(classifier_names):
    params = dict()
    if classifier_names == 'Knn':
        K = st.sidebar.slider('k', 1, 15)
        params['K'] = K
    elif classifier_names == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0,)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth # depth of every tree that grow in random forest 
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators # number of trees in random forest
    return params
params = add_paramater_ui(classifier_name)


# defining classifiers based on classifiers names and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'Knn':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                           max_depth=params['max_depth'], random_state=1234)
    return clf
clf = get_classifier(classifier_name, params)

# splitting the data into test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


acc = accuracy_score(y_test,y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc}')

# plotting all our our features on 2 dimenional plot using PCA
pca = PCA(2)
X_projected = pca.fit_transform(X)

# slicing the data into 0 and 1 dimensions 
X_0 = X_projected[:, 0]
X_1 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(X_0,X_1,
            c=y, alpha=0.8,
            cmap='viridis')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)