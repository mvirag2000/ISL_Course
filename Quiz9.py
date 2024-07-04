import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from sklearn.svm import SVC

def make_data(cases):
    covariance_matrix = np.eye(10)  # 10x10 identity matrix as the covariance matrix
    
    mean_vector_1 = np.zeros(10)  # 10-dimensional zero mean vector
    class1_x = np.random.multivariate_normal(mean_vector_1, covariance_matrix, int(cases/2))
    class1_y = np.zeros(class1_x.shape[0])

    mean_vector_2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    class2_x = np.random.multivariate_normal(mean_vector_2, covariance_matrix, int(cases/2))
    class2_y = np.ones(class2_x.shape[0])
    
    y = np.append(class1_y, class2_y)  
    x = np.vstack((class1_x, class2_x))
    
    shuffle = np.random.permutation(cases)
    x = x[shuffle]
    y = y[shuffle]
    return x, y

trials = 1000
test_cases = 1000
wrong = 0
for i in range(0, trials):
    x_train, y_train = make_data(100)

    model = LDA()
    # model = SVC(C=10, kernel='linear')
    # model = SVC(C=10, kernel='rbf')
    model.fit(x_train, y_train)

    x_test, y_test = make_data(test_cases)
    y_pred = model.predict(x_test)
    wrong += test_cases - accuracy_score(y_test, y_pred, normalize=False)

error_rate = wrong / (trials * test_cases)
print(error_rate)
