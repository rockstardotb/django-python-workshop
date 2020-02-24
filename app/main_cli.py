from app.utils import preprocessing as prep
from app.utils import logistic_regression as logreg

class Test_logreg(object):

    # Initialize your class attributes
    def __init__(self, data_path="", X=[], y=[], split=0.25, X_train=[], y_train=[],
                 X_test=[], y_test=[], sc_X=[]):

        self.data = data_path
        self.X = X
        self.y = y
        self.split = split
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.sc_X = sc_X

        self.prep_data()
        self.regressor = self.train()
        self.result = self.predict()

    def prep_data(self):
        self.X, self.y = prep.import_data(self.data, 4, 4, 2)
        self.X = self.X.astype(float)
        self.X_train, self.X_test, self.y_train, self.y_test = prep.create_sets(self.X, self.y, size=self.split)
        self.X_train, self.X_test, self.sc_X = prep.feature_scale(self.X_train, self.X_test)

    def train(self):
        return logreg.train(self.X_train,self.y_train,random_state=0)

    def predict(self):
        return self.regressor.predict(self.X_test)

    def accuracy(self):
        correct = 0
        for i in range(len(self.result)):
            if self.result[i] == self.y_test[i]:
                correct += 1

        return correct*100/len(self.X_test)

if __name__ == '__main__':

    test = Test_logreg(data_path='./data/Social_Network_Ads.csv')

    print('\nModel: Logistic Regression\nAccuracy: {}%\n'.format(test.accuracy()))

