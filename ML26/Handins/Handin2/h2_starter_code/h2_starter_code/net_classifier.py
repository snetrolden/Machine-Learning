import numpy as np

def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes 
    
    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc

def softmax(X):
    """ 
    You can take this from handin I
    Compute the softmax of each row of an input matrix (2D numpy array). 
    
    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate 
    
    More precisely this is what you must do.
    
    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max i.e (log sum exp x-max) + maxlog
    compute log of the softmax: x - logsum
    exponentiate that
    
    You can do all of it without for loops using numpys vectorized operations.

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)
    ### YOUR CODE HERE
    max_x = np.amax(X, axis=1, keepdims=True)
    logsum = np.log(np.sum(np.exp(X - max_x), axis=1, keepdims=True)) + max_x
    log_sm = X - logsum
    res = np.exp(log_sm)
    ### END CODE
    return res

def relu(x):
    """ Compute the relu activation function on every element of the input
    
        Args:
            x: np.array
        Returns:
            res: np.array same shape as x
        Beware of np.max and look at np.maximum
    """
    ### YOUR CODE HERE
    res = np.maximum(0,x)
    ### END CODE
    return res

def make_dict(W1, b1, W2, b2):
    """ Trivial helper function """
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def get_init_params(input_dim, hidden_size, output_size):
    """ Initializer function using Xavier/he et al Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

    Args:
      input_dim: int
      hidden_size: int
      output_size: int
    Returns:
       dict of randomly initialized parameter matrices.
    """
    W1 = np.random.normal(0, np.sqrt(2./(input_dim+hidden_size)), size=(input_dim, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.normal(0, np.sqrt(4./(hidden_size+output_size)), size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

  
class NetClassifier():
    
    def __init__(self):
        """ Trivial Init """
        self.params = None
        self.hist = None

    def predict(self, X, params=None):
        """ Compute class prediction for all data points in class X
        
        Args:
            X: np.array shape n, d
            params: dict of params to use (if none use stored params)
        Returns:
            np.array shape n, 1
        """
        if params is None:
            params = self.params
        pred = None
        ### YOUR CODE HERE
        W1 = params['W1'] 
        b1 = params['b1'] 
        W2 = params['W2'] 
        b2 = params['b2'] 

        Z1 = X @ W1 + b1
        a1 = relu(Z1)
        Z2 = a1 @ W2 + b2

        pred = np.argmax(Z2, axis=1)
        ### END CODE
        return pred
     
    def score(self, X, y, params=None):
        """ Compute accuracy of model on data X with labels y (mean 0-1 loss)
        
        Args:
            X: np.array shape n, d
            y: np.array shape n, 1
            params: dict of params to use (if none use stored params)

        Returns:
            acc: float, number of correct predictions divided by n. NOTE: This is accuracy, not in-sample error!
        """
        if params is None:
            params = self.params
        acc = None
        ### YOUR CODE HERE
        # accuracy = (1/n) * sum_i [h(x_i) == y_i]   (1 - mean 0-1 loss)
        acc = np.mean(self.predict(X, params) == y)
        ### END CODE
        return acc
    
    @staticmethod
    def cost_grad(X, y, params, c=0.0):
        """ Compute cost and gradient of neural net on data X with labels y using weight decay parameter c
        You should implement a forward pass and store the intermediate results 
        and then implement the backwards pass using the intermediate stored results
        
        Use the derivative for cost as a function for input to softmax as derived above
        
        Args:
            X: np.array shape n, self.input_size
            y: np.array shape n, 1
            params: dict with keys (W1, W2, b1, b2)
            c: float - weight decay parameter
            params: dict of params to use for the computation
        
        Returns 
            cost: scalar - average cross entropy cost with weight decay parameter c
            dict with keys
            d_w1: np.array shape w1.shape, entry d_w1[i, j] = \partial cost/ \partial W1[i, j]
            d_w2: np.array shape w2.shape, entry d_w2[i, j] = \partial cost/ \partial W2[i, j]
            d_b1: np.array shape b1.shape, entry d_b1[1, j] = \partial cost/ \partial b1[1, j]
            d_b2: np.array shape b2.shape, entry d_b2[1, j] = \partial cost/ \partial b2[1, j]
            
        """
        cost = 0
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        d_w1 = None
        d_w2 = None
        d_b1 = None
        d_b2 = None
        labels = one_in_k_encoding(y, W2.shape[1]) # shape n x k
                        
        ### YOUR CODE HERE - FORWARD PASS - compute cost with weight decay and store relevant values for backprop
        n = X.shape[0]

        # Layer 1: Z1 = X W1 + b1,  Relu
        # Layer 2: Z2 = a1 W2 + b2, Softmax  
        Z1 = X @ W1 + b1            # n x h
        a1 = relu(Z1)               # n x h
        Z2 = a1 @ W2 + b2           # n x k

       
        z_max = np.amax(Z2, axis=1, keepdims=True)
        log_sum = z_max + np.log(np.sum(np.exp(Z2 - z_max), axis=1, keepdims=True))
        log_probs = Z2 - log_sum    # n x k, ln softmax(Z2)
        probs = np.exp(log_probs)   # n x k, softmax(Z2)

        #Mean negative log likelihood
        nll = -np.sum(labels * log_probs) / n

        #Weight decay, biases NOT included
        decay = c * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        cost = nll + decay
        ### END CODE

        ### YOUR CODE HERE - BACKWARDS PASS - compute derivatives of all weights and bias, store them in d_w1, d_w2, d_b1, d_b2
    
        d_z2 = (probs - labels) / n                 # n x k
        d_w2 = a1.T @ d_z2 + 2 * c * W2             # h x k
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)  # 1 x k
        d_a1 = d_z2 @ W2.T                          # n x h

        # RelU
        d_z1 = d_a1 * (Z1 > 0)                      # n x h



        d_w1 = X.T @ d_z1 + 2 * c * W1              # d x h
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)  # 1 x h
        ### END CODE
        # the return signature
        return cost, {'d_w1': d_w1, 'd_w2': d_w2, 'd_b1': d_b1, 'd_b2': d_b2}
        
    def fit(self, X_train, y_train, X_val, y_val, init_params, batch_size=32, lr=0.1, c=1e-4, epochs=30):
        """ Run Mini-Batch Gradient Descent on data X, Y to minimize the in sample error for Neural Net classification
        Printing the performance every epoch is a good idea to see if the algorithm is working
    
        Args:
           X_train: numpy array shape (n, d) - the training data each row is a data point
           y_train: numpy array shape (n,) int - training target labels numbers in {0, 1,..., k-1}
           X_val: numpy array shape (n, d) - the validation data each row is a data point
           y_val: numpy array shape (n,) int - validation target labels numbers in {0, 1,..., k-1}
           init_params: dict - has initial setting of parameters
           lr: scalar - initial learning rate
           batch_size: scalar - size of mini-batch
           c: scalar - weight decay parameter 
           epochs: scalar - number of iterations through the data to use

        Sets: 
           params: dict with keys {W1, W2, b1, b2} parameters for neural net
        returns
           hist: dict:{keys: train_loss, train_acc, val_loss, val_acc} each an np.array of size epochs of the the given cost after every epoch
           loss is the NLL loss and acc is accuracy
        """
        
        W1 = init_params['W1']
        b1 = init_params['b1']
        W2 = init_params['W2']
        b2 = init_params['b2']
        hist = {
            'train_loss': None,
            'train_acc': None,
            'val_loss': None,
            'val_acc': None, 
        }

        
        ### YOUR CODE HERE
        # Copy for naming reasons
        params = {'W1': W1.copy(), 'b1': b1.copy(), 'W2': W2.copy(), 'b2': b2.copy()}
        n = X_train.shape[0]

        train_loss = np.zeros(epochs)
        train_acc = np.zeros(epochs)
        val_loss = np.zeros(epochs)
        val_acc = np.zeros(epochs)

        best_val_acc = -1.0
        best_params = None

        # Mini-batch SGD
        # See lecture 15 in folder (see comment in earlier commit)
        #   while more time:
        #     permute (x1,y1),...,(xn,yn) randomly
        #     for each next batch of B items:  w = w - lr * grad of the batch loss
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_perm = X_train[perm]
            y_perm = y_train[perm]
            for start in range(0, n, batch_size):
                X_batch = X_perm[start:start + batch_size]
                y_batch = y_perm[start:start + batch_size]
                _, grad = self.cost_grad(X_batch, y_batch, params, c=c)
                # Gradient step w = w - lr * dw
                params['W1'] -= lr * grad['d_w1']
                params['b1'] -= lr * grad['d_b1']
                params['W2'] -= lr * grad['d_w2']
                params['b2'] -= lr * grad['d_b2']


            
            train_loss[epoch], _ = self.cost_grad(X_train, y_train, params, c=0.0)
            val_loss[epoch], _ = self.cost_grad(X_val, y_val, params, c=0.0)
            train_acc[epoch] = self.score(X_train, y_train, params)
            val_acc[epoch] = self.score(X_val, y_val, params)

            # safekeep the best. see Lecture 16 (they ar numbered in the folder)
            
            if val_acc[epoch] > best_val_acc:
                best_val_acc = val_acc[epoch]
                best_params = {key: value.copy() for key, value in params.items()}

            # AI made print to help with ascetaining ressuklt.
            print('epoch {0:3d}: train loss {1:.4f} acc {2:.4f} | val loss {3:.4f} acc {4:.4f}'.format(
                epoch + 1, train_loss[epoch], train_acc[epoch], val_loss[epoch], val_acc[epoch]))

        hist = {'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc}
        self.params = best_params
        self.hist = hist
        ### END CODE
        # hist dict should look like this with something different than none
        #hist = {'train_loss': None, 'train_acc': None, 'val_loss': None, 'val_acc': None}
        ## self.params should look like this with something better than none, i.e. the best parameters found.
        # self.params = {'W1': None, 'b1': None, 'W2': None, 'b2': None}
        return hist
        

def numerical_grad_check(f, x, key):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-5
    # d = x.shape[0]
    cost, grad = f(x)
    grad = grad[key]
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:    
        dim = it.multi_index    
        print(dim)
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h 
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus-cminus)/(2*h)
        # print('cplus cminus', cplus, cminus, cplus-cminus)
        # print('dim, grad, num_grad, grad-num_grad', dim, grad[dim], num_grad, grad[dim]-num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim, num_grad, grad[dim])
        it.iternext()

def test_grad():
    stars = '*'*5
    print(stars, 'Testing  Cost and Gradient Together')
    input_dim = 7
    hidden_size = 1
    output_size = 3
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)

    nc = NetClassifier()
    X = np.random.randn(7, input_dim)
    y = np.array([0, 1, 2, 0, 1, 2, 0])

    f = lambda z: nc.cost_grad(X, y, params, c=1.0)
    print('\n', stars, 'Test Cost and Gradient of b2', stars)
    numerical_grad_check(f, params['b2'], 'd_b2')
    print(stars, 'Test Success', stars)
    
    print('\n', stars, 'Test Cost and Gradient of w2', stars)
    numerical_grad_check(f, params['W2'], 'd_w2')
    print('Test Success')
    
    print('\n', stars, 'Test Cost and Gradient of b1', stars)
    numerical_grad_check(f, params['b1'], 'd_b1')
    print('Test Success')
    
    print('\n', stars, 'Test Cost and Gradient of w1', stars)
    numerical_grad_check(f, params['W1'], 'd_w1')
    print('Test Success')

if __name__ == '__main__':
    input_dim = 3
    hidden_size = 5
    output_size = 4
    batch_size = 7
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)
    X = np.random.randn(batch_size, input_dim)
    Y = np.array([0, 1, 2, 0, 1, 2, 0])
    nc.cost_grad(X, Y, params, c=0)
    test_grad()
