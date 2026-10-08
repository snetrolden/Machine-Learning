import numpy as np
from h1_util import numerical_grad_check


## IMPLEMENTED
def logistic(z):
    """ 
    Helper function
    Computes the logistic function 1/(1+e^{-x}) to each entry in input vector z.
    
    np.exp may come in handy
    Args:
        z: numpy array shape (d,) 
    Returns:
       logi: numpy array shape (d,) each entry transformed by the logistic function 
    """
    logi = np.zeros(z.shape)
    ### YOUR CODE HERE

    # Iterate over the size of z and apply the function to each element.
    for i in range(len(z)):
        logi[i] = 1/(1+np.exp(-z[i]))

    ### END CODE
    assert logi.shape == z.shape
    return logi


class LogisticRegressionClassifier():

    def __init__(self):
        self.w = None

    ## NOT IMPLEMNETED TODO
    def cost_grad(self, X, y, w):
        """
        Compute the average negative log likelihood and gradient under the logistic regression model 
        using data X, targets y, weight vector w 
        
        np.log, np.sum, np.choose, np.dot may be useful here
        Args:
           X: np.array shape (n,d) float - Features 
           y: np.array shape (n,)  int - Labels 
           w: np.array shape (d,)  float - Initial parameter vector

        Returns:
           cost: scalar: the average negative log likelihood for logistic regression with data X, y 
           grad: np.array shape(d, ) gradient of the average negative log likelihood at w 
        """
        cost = 0
        grad = np.zeros(w.shape)
    
        ### YOUR CODE HERE
        n = X.shape[0]
        # Margin: s_i = y_i * (x_i . w)
        exponent = y * np.dot(X, w)
        
        # Average negative log-likelihood
        # numpy has np.logaddexp(0, -s) that computes ln(1 + exp(-s)) with less overflow (numerical stability) #trust
        cost = np.mean(np.logaddexp(0, -exponent))
        
        # Gradient:
        # logistic(-exponent), let arg be negative, to get a positive output 
        # transpose X to get correct dimensions
        grad = -np.dot(X.T, y * logistic(-exponent))
        ### END CODE
        assert grad.shape == w.shape
        return cost, grad


    ## NOT IMPLEMENTED TODO
    def fit(self, X, y, w=None, lr=0.1, batch_size=16, epochs=10):
        """
        Run mini-batch stochastic Gradient Descent for logistic regression 
        use batch_size data points to compute gradient in each step.
    
        The function np.random.permutation may prove useful for shuffling the data before each epoch
        It is wise to print the performance of your algorithm at least after every epoch to see if progress is being made.
        Remember the stochastic nature of the algorithm may give fluctuations in the cost as iterations increase.

        Args:
           X: np.array shape (n,d) dtype float32 - Features 
           y: np.array shape (n,) dtype int32 - Labels 
           w: np.array shape (d,) dtype float32 - Initial parameter vector
           lr: scalar - learning rate for gradient descent
           batch_size: number of elements to use in minibatch
           epochs: Number of scans through the data

        sets: 
           w: numpy array shape (d,) learned weight vector w
           history: list/np.array len epochs - value of loss function (in-sample error) after every epoch. Used for plotting
        """
        if w is None: w = np.zeros(X.shape[1])
        history = []        
        ### YOUR CODE HERE 
        n = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data before each epoch
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent thing
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                _, grad = self.cost_grad(X_batch, y_batch, w) # not interested in the cost here
                w = w - lr * grad
            
            # remember cost for each epoch
            epoch_cost, _ = self.cost_grad(X, y, w) # cost here
            history.append(epoch_cost)
            print(f"Epoch {epoch + 1}/{epochs} - Cost: {epoch_cost:.4f}") #check status after each epoch
        ### END CODE
        self.w = w
        self.history = history


    ## NOT IMEPLEMTED TODO
    def predict(self, X):
        """ Classify each data element in X.

        Args:
            X: np.array shape (n,d) dtype float - Features   
        
        Returns: 
           p: numpy array shape (n, ) dtype int32, class predictions on X (-1, 1). NOTE: We want a class here, 
           not a probability between 0 and 1. You should thus return the most likely class!

        """
        out = np.ones(X.shape[0])
        ### YOUR CODE HERE
        w = self.w 
        sigmoid = logistic(np.dot(X, w)) ## Helper function to compress value into a probability between 0 and 1
        out[sigmoid < 0.5] = -1 # everyhting under 0.5 after the sigmoid func is classified as -1 and everything else 1
        ### END CODE
        return out

    ## NOT IMPLEMENTED TODO
    def score(self, X, y):
        """ Compute model accuracy  on Data X with labels y

        Args:
            X: np.array shape (n,d) dtype float - Features 
            y: np.array shape (n,) dtype int - Labels 

        Returns: 
           s: float, number of correct predictions divided by n. NOTE: This is accuracy, not in-sample error!

        """
        s = 0
        ### YOUR CODE HERE
        # get predictions and compare with true labels to compute accuracy
        s = np.mean(self.predict(X) == y)
        ### END CODE
        return s
        

    
def test_logistic():
    print('*'*5, 'Testing logistic function')
    a = np.array([0, 1, 2, 3])
    lg = logistic(a)
    target = np.array([ 0.5, 0.73105858, 0.88079708, 0.95257413])
    assert np.allclose(lg, target), 'Logistic Mismatch Expected {0} - Got {1}'.format(target, lg)
    print('Test Success!')

    
def test_cost():
    print('*'*5, 'Testing Cost Function')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [3, 2]])
    y = np.array([-1, -1, 1], dtype='int64')
    w = np.array([0.0, 0.0])
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    cost,_ = lr.cost_grad(X, y, w)
    target = -np.log(0.5)
    assert np.allclose(cost, target), 'Cost Function Error:  Expected {0} - Got {1}'.format(target, cost)
    print('Test Success')

    
def test_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 3.0]])    
    w = np.array([0.0, 0.0])
    y = np.array([-1, -1, 1]).astype('int64')
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    f = lambda z: lr.cost_grad(X, y, w=z)
    numerical_grad_check(f, w)
    print('Test Success')


    
if __name__ == '__main__':
    test_logistic()
    test_cost()
    test_grad()
    
    
