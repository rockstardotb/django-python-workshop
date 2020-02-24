from sklearn.linear_model import LogisticRegression as logreg

# fit data to logistic curve
def train(X_train,y_train,random_state=0):
    regressor = logreg(random_state=random_state)
    regressor.fit(X_train,y_train)
    return regressor

# Make a prediction using the trained model
def predict(X_test, regressor):
    y_pred = regressor.predict(X_test)

    return y_pred

# Plot expected vs predicted
def visualize(X_train, y_train, regressor):
    import matplotlib.pyplot as plt
    import numpy as np

    X_grid = np.arange(min(X_train),max(X_train),0.001)
    X_grid = X_grid.reshape((len(X_grid),1))
    plt.scatter(X_train, y_train, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
    plt.title('Y vs X (Training Set)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Create a confusion matrix [[correct,incorrect],[incorrect,correct]]
def confusion(X_test,y_test, X_train, y_train, regressor):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    from sklearn.metrics import confusion_matrix as cmat

    y_pred = predict(X_test,regressor)
    cm = cmat(y_test,y_pred)

    X_set, y_set = X_train,y_train

    X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),
                         np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
    plt.contour(X1,X2,regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('red','green')))
    plt.xlim(X1.min(),X1.max())
    plt.ylim(X2.min(),X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0], X_set[y_set==j,1],
                    c=ListedColormap(('red','green'))(i), label=j)
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    return cm


