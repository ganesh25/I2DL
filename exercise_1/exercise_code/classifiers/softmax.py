"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    ############################################################################
    #                           CROSS ENTROPY LOSS                             #
    ############################################################################
    
    num_classes = W.shape[1]
    num_examples = X.shape[0]
    
    for i in range(num_examples):
        
        # f(xi,W): xi*W -> scores vector
        f = np.dot(X[i], W)    
               
        # since the exponentiation might result in large values, we need to   #
        # take care that the calculation is numerically stable.               #
        # shift the values of f so that the highest value is zero.            #
        f = f - f.max()              
        
        # exponentiation of score of correct class                            
        f_i = np.exp(f[y[i]])
        
        # exponential sum of scores of all classes                            
        f_j = np.sum(np.exp(f))
        
        # normalization of data_loss for each example                                                     
        loss = loss + (- np.log(f_i / f_j))
   

        ############################################################################
        #                                GRADIENT                                  #
        ############################################################################
        
        # for correct class
        dW[:, y[i]] += -(f_j - f_i) / f_j * X[i]
        
        for j in range(num_classes):
            
            if j == y[i]:
                continue
            
            else:
                dW[:, j] += (np.exp(f[j]) / f_j) * X[i]
                  

    # data loss
    loss = loss / num_examples
    
    # total loss = data_loss + regularization_loss
    # write 0.5 * reg in order to simplify the expression of gradient
    loss = loss + (0.5 * reg * np.sum(W *W))
    
    dW = dW / num_examples
    
    dW = dW + (reg * W)
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    ############################################################################
    #                           CROSS ENTROPY LOSS                             #
    ############################################################################
    
    num_classes = W.shape[1]
    num_examples = X.shape[0]
    
    # f(xi,W): xi*W -> scores vector
    f = X.dot(W)
    
    # since the exponentiation might result in large values, we need to   #
    # take care that the calculation is numerically stable.               #
    # shift the values of f so that the highest value is zero.            #
    f = f - f.max()
    
    # exponential of scores of all classes 
    f = np.exp(f)
    
    # exponential sum of scores of all classes
    f_sum = np.sum(f, axis = 1)
    
    # 1D array of just the probabilities assigned to the correct classes  #
    # for each example (non-normalized probability)                       #
    f_i = f[range(num_examples), y]
    
    # normalization of probabilities for each example (data_loss) 
    loss = f_i / f_sum
    
    # total loss = data_loss + regularization_loss
    loss = (np.sum(-np.log(loss)) / num_examples) + (0.5 * reg * np.sum(W * W))
    
    
    ############################################################################
    #                                GRADIENT                                  #
    ############################################################################
    
    # normalized scores vector
    f_norm = np.divide(f, f_sum.reshape(num_examples,1))
    
    # for correct class
    f_norm[range(num_examples), y] = -(f_sum - f_i) / f_sum
    
    dW = np.transpose(X).dot(f_norm)
    
    dW = dW / num_examples
    
    dW = dW + (reg * W)
    
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [7e-6, 7e-7, 7e-8, 7e-9]
    regularization_strengths = [1e4, 1e3, 1e2, 1e1]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    
    # create an instance of SoftmaxClassifier
    smax_classy = SoftmaxClassifier()
    
    # training a classifier on the training set for each combination of hyper-
    # parameters
    for learn_rate in learning_rates:
        for reg_strength in regularization_strengths:
            loss_history = smax_classy.train(X_train, y_train, learning_rate = learn_rate, reg = reg_strength, num_iters = 1500)
            
            # predict class labels on the training set
            y_predic_trainset = smax_classy.predict(X_train)
            
            # accuracy of the predicted class labels for training set
            acc_trainset = np.mean(y_predic_trainset == y_train)
            
            
            
            # predict class labels on the validation set
            y_predic_valset = smax_classy.predict(X_val)
            
            # accuracy of the predicted class labels for validation set
            acc_valset = np.mean(y_predic_valset == y_val)
            
            
            # storing the accuracies of training set and validation set in results dictionary
            results[(learn_rate, reg_strength)] = (acc_trainset, acc_valset)
            
            # update the best choice
            if acc_valset > best_val:
                best_val = acc_valset
                best_softmax = smax_classy  
    
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
