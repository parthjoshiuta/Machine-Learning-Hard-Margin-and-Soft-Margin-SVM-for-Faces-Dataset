import matplotlib as mpl
mpl.use('TkAgg')
import os
import numpy as np
import cvxopt
import scipy.misc
from sklearn.utils import shuffle
from sklearn import cross_validation

class Support_Vector_Machine():

    def __init__(self):
        # Defining global variables
        self.c = 100.0
        self.lm = None
        self.sv_X = None
        self.sv_y = None
        self.b = None
        self.w = None

    def train(self, x, y):
        # Extracting sample and feature lengths
        sample_len, feature_len = x.shape

        # Generating Gramian Matrix
        M = np.zeros((sample_len, sample_len))
        for i in range(sample_len):
            for j in range(sample_len):
                M[i, j] = np.dot(x[i], x[j])

        # Calculating values of P, q, A, b, G, h
        P = cvxopt.matrix(np.outer(y, y) * M)
        q = cvxopt.matrix(np.ones(sample_len) * -1)
        A = cvxopt.matrix(y, (1, sample_len), 'd')
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(sample_len) * -1), np.identity(sample_len))))
        h = cvxopt.matrix(np.hstack((np.zeros(sample_len), np.ones(sample_len) * self.c)))

        # Solving quadratic equation
        # You can set 'show_progress' to True to see cvxopt output
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        lm = np.ravel(sol['x'])

        # Determining support vectors
        y = np.asarray(y)
        sv = lm > 0.0e-7
        index = np.arange(len(lm))[sv]
        self.lm = lm[sv]
        self.sv_X = x[sv]
        self.sv_y = y[sv]

        # Calculating bias
        self.b = 0.0
        for i in range(len(self.lm)):
            self.b = self.b + self.sv_y[i] - np.sum(self.lm * self.sv_y * M[index[i], sv])
        self.b /= len(self.lm)

        # Calculating weights
        self.w = np.zeros(feature_len)
        for i in range(len(self.lm)):
            self.w += self.lm[i] * self.sv_y[i] * self.sv_X[i]

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)

if __name__ == "__main__":
    def img_input(resize=False):
        X, y = [], []
        for path, direc, docs in os.walk("orl_faces"):
            direc.sort()
            # Iterating through each subject
            for subject in direc:
                files = os.listdir(path + '/' + subject)
                for file in files:
                    img = scipy.misc.imread(path + '/' + subject + '/' + file).astype(np.float32)
                    if resize:
                        img = scipy.misc.imresize(img, (56, 46)).astype(np.float32)
                    X.append(img.reshape(-1))
                    y.append(int(subject[1:]))
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y

    def cross_val(X, y, DR=None, Alg=None, All=False):
        X, y = shuffle(X, y)
        kf = cross_validation.KFold(len(y), n_folds=5)
        avg_acc_svm = []
        fld = 1
        for train, test in kf:
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            if DR is None and Alg == 'SVM' or All:
                print('\nRunning Fold', fld, 'for SVM')
                # Defining svm model
                svm = Support_Vector_Machine()
                y_train_ovr = [None] * len(y_train)
                y_test_ovr = [None] * len(y_test)
                accuracies = 0
                print('Running SVM. Please wait...')
                for i in range(1, 41):
                    # Setting the selected class as '1' and rest as '-1' depicting the One vs Rest classification.
                    for j in range(0, 320):
                        if y_train[j] == (i):
                            y_train_ovr[j] = 1
                        else:
                            y_train_ovr[j] = -1
                    for j in range(0, 80):
                        if y_test[j] == (i):
                            y_test_ovr[j] = 1
                        else:
                            y_test_ovr[j] = -1
                    # Taking Set_A as training set and Set_B for testing
                    svm.train(X_train, y_train_ovr)
                    predict_class = svm.predict(X_test)
                    c = np.sum(predict_class == y_test_ovr)
                    accuracies += float(c) / len(predict_class) * 100
                accuracy = accuracies / 40
                print('Accuracy is ', accuracy)
                avg_acc_svm.append(accuracy)
                if not All:
                    fld += 1
            if All:
                fld += 1
            print('\n')
        if DR is None and Alg == 'SVM' or All:
            print('Average accuracy for SVM ', sum(avg_acc_svm) / 5.0, '\n')

    def task_selector():
        print("Running SVM\n")
        X, y = img_input()
        cross_val(X, y, DR=None, Alg='SVM')

    task_selector()