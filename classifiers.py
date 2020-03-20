# Template by Bruce Maxwell
# Spring 2015
# Some unremembered amount of work done by Evan Savillo 2016
#
# Classifier class and child definitions


import sys
import data
import analysis as an
import scipy.cluster.vq as vq
import numpy as np

class Classifier:

    def __init__(self, type):
        '''The parent Classifier class stores only a single field: the type of
        the classifier.  A string makes the most sense.

        '''
        self._type = type

    def type(self, newtype = None):
        '''Set or get the type with this function'''
        if newtype != None:
            self._type = newtype
        return self._type

    def confusion_matrix(self, truecats, classcats):
        '''Takes in two Nx1 matrices of zero-index numeric categories and
        computes the confusion matrix. The rows represent true
        categories, and the columns represent the classifier output.

        '''
        n = len(truecats)
        A = np.bincount(n * (truecats - 1) + (classcats -1), minlength=n*n).reshape(n, n)

        return A

    def confusion_matrix_str(self, cmtx):
        '''Takes in a confusion matrix and returns a string suitable for printing.'''
        s = ''
        for row in cmtx:
            for element in row:
                s += str(element).strip('[]').center(4)
            s += '\n'

        return s

    def __str__(self):
        '''Converts a classifier object to a string.  Prints out the type.'''
        return str(self._type)



class NaiveBayes(Classifier):
    '''NaiveBayes implements a simple NaiveBayes classifier using a
    Gaussian distribution as the pdf.

    '''

    def __init__(self, dataObj=None, headers=[], categories=None):
        '''Takes in a Data object with N points, a set of F headers, and a
        matrix of categories, one category label for each data point.'''

        # call the parent init with the type
        Classifier.__init__(self, 'Naive Bayes Classifier')
        
        # store the headers used for classification
        # number of classes and number of features
        # original class labels
        # unique data for the Naive Bayes: means, variances, scales
        # if given data,
            # call the build function
        self.headers = headers
        self.num_features = len(headers)

        if dataObj is not None:
            if self.num_features < 1:
                self.headers = dataObj.get_headers()
                self.num_features = len(headers)

            self.build(dataObj.get_data(self.headers).T, categories)

    def build(self, A, categories):
        '''Builds the classifier given the data points in A and the categories'''

        # figure out how many categories there are and get the mapping (np.unique)
        self.class_labels, map2, mapping = np.unique(np.array(categories.T),
                                                     return_index=True, return_inverse=True)

        # create the matrices for the means, vars, and scales
        # the output matrices will be categories (C) x features (F)
        # compute the means/vars/scales for each class
        if len(map2) < 2:
            means = A[map2[0]:].mean(0)[0]
            variances = A[map2[0]:].var(0)[0]
            scales = 1/np.sqrt(2*np.pi*variances[0])
        else:
            means = A[map2[0]:map2[1]].mean(0)[0]
            variances = A[map2[0]:map2[1]].var(0)[0]
            scales = 1/np.sqrt(2*np.pi*variances[0])

            for i in range(1,len(map2)-1):
                means = np.concatenate( (means, A[map2[i]:map2[i+1]].mean(0)[0]) )
                variances = np.concatenate( (variances, A[map2[i]:map2[i+1]].var(0)[0]) )
                scales = np.concatenate( (scales, 1/np.sqrt(2*np.pi*variances[i])) )

            means = np.concatenate( (means, A[map2[-1]:].mean(0)[0]) )
            variances = np.concatenate( (variances, A[map2[-1]:].var(0)[0]) )
            scales = np.concatenate( (scales, 1/np.sqrt(2*np.pi*variances[-1])) )

        self.class_means = means
        self.class_vars = variances
        self.class_scales = scales

        # store any other necessary information: # of classes, # of features, original labels
        self.num_classes = len(self.class_labels)

        return

    def classify(self, A, return_likelihoods=False):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_likelihoods
        is True, it also returns the NxC likelihood matrix.

        '''

        # error check to see if A has the same number of columns as
        # the class means
        if len(A.T) != len(self.class_means.T):
            print 'Luks lyke ya got en errah, bru. Try ta .T somefin, ya?'
            quit()
        
        # make a matrix that is N x C to store the probability of each
        # class for each data point
        P = '' # a matrix of zeros that is N (rows of A) x C (number of classes)
        for i in range(len(A)):
            s = '0,' * self.num_classes
            P += s.rstrip(',') + ';'
        P = P.rstrip(';')

        P = np.matrix(P, dtype=np.float64)

        # calculate the probabilities by looping over the classes
        #  with numpy-fu you can do this in one line inside a for loop

        for x in range(len(A)):
            for i in range(len(self.class_means)):
                a = self.class_scales[i]
                b = np.exp(-np.square(A[x]-self.class_means[i])/(2*self.class_vars[i]))
                c = np.concatenate((a,b))
                P[x,i] = np.prod(c)

        # calculate the most likely class for each data point
        cats = P.argmax(1) # take the argmax of P along axis 1


        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_likelihoods:
            return cats, labels, P

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nNaive Bayes Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Mean  : ' + str(self.class_means[i,:]) + "\n"
            s += 'Var   : ' + str(self.class_vars[i,:]) + "\n"
            s += 'Scales: ' + str(self.class_scales[i,:]) + "\n"

        s += "\n"
        return s
        
    def write(self, filename):
        '''Writes the Bayes classifier to a file.'''
        # extension
        return

    def read(self, filename):
        '''Reads in the Bayes classifier from the file'''
        # extension
        return

    
class KNN(Classifier):
    ''' Take in a Data object with N points, a set of F headers, and a
    matrix of categories, with one category label for each data point. '''
    def __init__(self, dataObj=None, headers=[], categories=None, K=None):

        # call the parent init with the type
        Classifier.__init__(self, 'KNN Classifier')
        
        # store the headers used for classification
        self.headers = headers
        # number of classes and number of features
        self.num_classes = None
        self.num_features = len(headers)
        # original class labels
        self.class_labels = None

        # unique data for the KNN classifier: list of exemplars (matrices)
        self.exemplars = []

        # if given data,
            # call the build function
        if dataObj is not None:
            if self.num_features < 1:
                self.headers = dataObj.get_headers()
                self.num_features = len(headers)

            self.build(dataObj.get_data(self.headers).T, categories)


    '''
    Write the code for building a KNN classifier: the __init__ and build methods of the KNN class.
    Store the example data points for a class in a matrix, with each point as a row.
    Store the set of matrices, one for each class, in a list.

    The default classifier should take in the training data and store it as a set of C
    (number of categories) matrices, where each matrix is the set of points in category i.

    If the build method is given a value for the parameter K, however, then it should execute
     K-means clustering on each category. In other words, for each category, execute K-means
     for that category, creating K exemplar data points (the codebook output of the
     K-means clustering). Store only the codebook returned by K-means for use by the KNN classifier.
    '''
    def build(self, A, categories, K=None):
        '''Builds the classifier give the data points in A and the categories'''

        # figure out how many categories there are and get the mapping (np.unique)
        idx = np.argsort(categories)
        A = A[idx]
        categories = categories[idx]

        self.class_labels, occ = np.unique(categories.A1, return_index=True)

        # for each category i, build the set of exemplars
        for i in range( len(self.class_labels) ):
            if K is None:
                # append to exemplars a matrix with all of the rows of A where the category/mapping is i
                if i == len(self.class_labels)-1:
                    self.exemplars.append( A[occ[-1]:] )
                else:
                    self.exemplars.append( A[occ[i]:occ[i+1]] )

            else:
                # run K-means on the rows of A where the category/mapping is i
                # append the codebook to the exemplars
                if i == len(self.class_labels)-1:
                    self.exemplars.append( vq.kmeans(A[occ[-1]:], K)[0] )
                else:
                    self.exemplars.append( vq.kmeans(A[occ[i]:occ[i+1]], K)[0] )

        # store any other necessary information: # of classes, # of features, original labels
        self.num_classes = len(self.class_labels)

        return

    def classify(self, A, K=3, return_distances=False):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_distances is
        True, it also returns the NxC distance matrix.

        The parameter K specifies how many neighbors to use in the
        distance computation. The default is three.'''

        # error check to see if A has the same number of columns as the class means
        

        # make a matrix that is N x C to store the distance to each class for each data point
        D = '' # a matrix of zeros that is N (rows of A) x C (number of classes)
        
        # for each class i
            # make a temporary matrix that is N x M where M is the number of examplars (rows in exemplars[i])
            # calculate the distance from each point in A to each point in exemplar matrix i (for loop)
            # sort the distances by row
            # sum the first K columns
            # this is the distance to the first class

        # calculate the most likely class for each data point
        cats = '' # take the argmin of D along axis 1

        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_distances:
            return cats, labels, D

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nKNN Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Number of Exemplars: %d\n' % (self.exemplars[i].shape[0])
            s += 'Mean of Exemplars  :' + str(np.mean(self.exemplars[i], axis=0)) + "\n"

        s += "\n"
        return s


    def write(self, filename):
        '''Writes the KNN classifier to a file.'''
        # extension
        return

    def read(self, filename):
        '''Reads in the KNN classifier from the file'''
        # extension
        return
    

