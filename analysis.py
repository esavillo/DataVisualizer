# Evan Savillo
# 04/11/16

import data
import numpy as np
import scipy.stats as ss
import scipy.cluster.vq as vq
import random
import scipy.spatial.distance


'''
Takes in a list of column headers and the Data object and returns a list of 2-element
lists with the minimum and maximum values for each column.
'''
def data_range(data, headers):
    ranges = []

    cols = data.get_data(headers)
    mins = cols.min(1)
    maxs = cols.max(1)

    for i in range( mins.size ):
        ranges += [ [mins[i,0], maxs[i,0]] ]

    return ranges

'''
Takes in a list of column headers and the Data object and returns a list of the mean
values for each column.
'''
def mean(data, headers):
    means = []

    cols = data.get_data(headers)
    cols = cols.mean(1)

    for i in range( cols.size ):
        means += [ cols[i,0] ]

    return means

'''
Takes in a list of column headers and the Data object and returns a list of the
standard deviation for each specified column.
'''
def stdev(data, headers):
    stdevs = []

    cols = data.get_data(headers)
    cols = cols.std(1)

    for i in range( cols.size ):
        stdevs += [ cols[i,0] ]

    return stdevs

'''
Takes in a list of column headers and the Data object and returns a matrix with
each column normalized so its minimum value is mapped to zero and its
maximum value is mapped to 1.
'''
def normalize_columns_separately(data, headers):
    ranges = data_range(data, headers)
    cols = data.get_data(headers)

    # subtracts min value
    t = ''
    for i in range( len(headers) ):
        m = ''
        m += str(ranges[i][0]) + ' '
        m *= cols[i].size
        t += m[:-1] + '; '
    t = t[:-2]
    m1 = np.matrix(t)

    cols = cols - m1

    # scales to [0,1]
    for i in range( len(headers) ):
        drange = data_range(data,headers)
        cols[i] = cols[i]/(drange[i][1]-drange[i][0])

    return cols

'''
Takes in a list of column headers and the Data object and returns a matrix with
each entry normalized so that the minimum value (of all the data in this set of columns)
is mapped to zero and its maximum value is mapped to 1.
'''
def normalize_columns_together(data, headers):
    ranges = data_range(data,headers)
    cols = data.get_data(headers)

    minn = ranges[0][0]
    maxn = ranges[0][1]
    for ranje in ranges:
        minn = min(minn,ranje[0])
        maxn = max(maxn,ranje[1])

    # subtracts min value
    t = ''
    for i in range( len(headers) ):
        m = ''
        m += str(minn) + ' '
        m *= cols[i].size

        t += m[:-1] + '; '
    t = t[:-2]
    m1 = np.matrix(t)

    cols = cols - m1

    # scales to [0,1]
    for i in range( len(headers) ):
        cols[i] = cols[i]/(maxn-minn)

    return cols

'''
Calculates the median
'''
def median(data, headers):
    meds = []

    cols = data.get_data(headers)
    cols = np.sort(cols)

    for col in cols:
        if cols[0].size % 2 == 1:
            meds += col[0,col.size/2]
        else:
            meds += [( col[0,col.size/2] + col[0,col.size/2-1] ) / 2]

    return meds

'''
Takes in the data set, a list of headers for the independent variables,
and a single header (not in a list) for the dependent variable. Implements
linear regression for one or more independent variables.
'''
def linear_regression(data_o, i_headers, d_header):
    # assign to y the column of data for the dependent variable
    # assign to A the columns of data for the independent variables
    #    It's best if both y and A are numpy matrices
    # add a column of 1's to A to represent the constant term in the
    #    regression equation.  Remember, this is just y = mx + b (even
    #    if m and x are vectors).

    y = data_o.get_data([d_header]).T
    A = data_o.get_data(i_headers).T
    A = np.concatenate( (A, data.generateColumn(len(A),1).T), 1 )

    # assign to AAinv the result of calling numpy.linalg.inv( np.dot(A.T, A))
    #    The matrix A.T * A is the covariance matrix of the independent
    #    data, and we will use it for computing the standard error of the
    #    linear regression fit below.

    AAinv = np.linalg.inv( np.dot(A.T, A) )

    # assign to x the result of calling numpy.linalg.lstsq( A, y )
    #    This solves the equation y = Ab, where A is a matrix of the
    #    independent data, b is the set of unknowns as a column vector,
    #    and y is the dependent column of data.  The return value x
    #    contains the solution for b.

    x = np.linalg.lstsq(A, y)

    # assign to b the first element of x.
    #    This is the solution that provides the best fit regression
    # assign to N the number of data points (rows in y)
    # assign to C the number of coefficients (rows in b)
    # assign to df_e the value N-C,
    #    This is the number of degrees of freedom of the error
    # assign to df_r the value C-1
    #    This is the number of degrees of freedom of the model fit
    #    It means if you have C-1 of the values of b you can find the last one.

    b = x[0]
    N = len(y)
    C = len(b)
    df_e = N-C
    df_r = C-1

    # assign to error, the error of the model prediction.  Do this by
    #    taking the difference between the value to be predicted and
    #    the prediction. These are the vertical differences between the
    #    regression line and the data.
    #    y - numpy.dot(A, b)

    error = y - np.dot(A, b)

    # assign to sse, the sum squared error, which is the sum of the
    #    squares of the errors computed in the prior step, divided by the
    #    number of degrees of freedom of the error.  The result is a 1x1 matrix.
    #    numpy.dot(error.T, error) / df_e

    sse = np.dot(error.T, error) / df_e

    # assign to stderr, the standard error, which is the square root
    #    of the diagonals of the sum-squared error multiplied by the
    #    inverse covariance matrix of the data. This will be a Cx1 vector.
    #    numpy.sqrt( numpy.diagonal( sse[0, 0] * AAinv ) )

    stderr = np.sqrt( np.diagonal( sse[0, 0] * AAinv ) )

    # assign to t, the t-statistic for each independent variable by dividing
    #    each coefficient of the fit by the standard error.
    #    t = b.T / stderr

    t = b.T / stderr

    # assign to p, the probability of the coefficient indicating a
    #    random relationship (slope = 0). To do this we use the
    #    cumulative distribution function of the student-t distribution.
    #    Multiply by 2 to get the 2-sided tail.
    #    2*(1 - scipy.stats.t.cdf(abs(t), df_e))

    p = 2*(1 - ss.t.cdf(abs(t), df_e))

    # assign to r2, the r^2 coefficient indicating the quality of the fit.
    #    1 - error.var() / y.var()

    r2 = 1 - error.var() / y.var()

    # Return the values of the fit (b), the sum-squared error, the
    #     R^2 fit quality, the t-statistic, and the probability of a
    #     random relationship.

    return b, sse[0,0], r2, t, p

'''
Takes in a list of column headers and returns a PCAData object
optional pre-normalize parameter
'''
def pca(d, headers, normalize=True):
    if normalize:
        A = normalize_columns_separately(d, headers).T
    else:
        A = d.get_data(headers).T

    C = np.cov(A, rowvar=False)

    W, V  = np.linalg.eig(C)
    W = np.real(W)
    V = np.real(V)

    idx = W.argsort()
    idx = idx[::-1]
    W = W[idx]
    V = V[:,idx].T

    m = A.mean(0)

    D = A - m
    projd_data = V * D.T

    return data.PCAData(headers, projd_data.T, W, V, m)

'''
Takes in a Data object, a set of headers, and the number of clusters to create
Computes and returns the codebook, codes, and representation error.
'''
def kmeans_numpy(d, headers, K, whiten=True, inputCB=None):
    A = d.get_data(headers).T

    if whiten is True:
        W = vq.whiten(A)
    else:
        W = A

    if inputCB is None:
        codebook, bookerror = vq.kmeans(W, K)
    else:
        codebook = inputCB

    codes, error = vq.vq(W, codebook)

    return codebook, codes, error

# O(iterations*clusters*instances*dimensions)

# where c_i is the cluster mean in the original data space for dimension i,
# cw_i is the whitened cluster mean, m_i is the data mean
# and s_i is the standard deviation.
# *** c_i = c_w*s_i + m_i *** #
'''
Takes in a Data object, a set of headers, and the number of clusters to create
Computes and returns the codebook, codes and representation errors.
If given an Nx1 matrix of categories, it uses the category labels
to calculate the initial cluster means.
'''
def kmeans(d, headers, K, whiten=True, categories=None, metric='sum2ed'):
    A = d.get_data(headers).T

    if whiten is True:
        W = vq.whiten(A)
    else:
        W = A

    codebook = kmeans_init(W, K, categories)
    codebook, codes, errors = kmeans_algorithm(W, codebook, metric)

    return codebook, codes, errors

def kmeans_algorithm(A, means, metric):
    # set up some useful constants
    MIN_CHANGE = 1e-7
    MAX_ITERATIONS = 100
    D = means.shape[1]
    K = means.shape[0]
    N = A.shape[0]

    # iterate no more than MAX_ITERATIONS
    for i in range(MAX_ITERATIONS):
        # calculate the codes
        codes, errors = kmeans_classify( A, means, metric )

        # calculate the new means
        newmeans = np.zeros_like( means )
        counts = np.zeros( (K, 1) )
        for j in range(N):
            newmeans[codes[j,0],:] += A[j,:]
            counts[codes[j,0],0] += 1.0

        # finish calculating the means, taking into account possible zero counts
        for j in range(K):
            if counts[j,0] > 0.0:
                newmeans[j,:] /= counts[j, 0]
            else:
                newmeans[j,:] = A[random.randint(0,A.shape[0]),:]

        # test if the change is small enough
        diff = np.sum(np.square(means - newmeans))
        means = newmeans
        if diff < MIN_CHANGE:
            break

    # call classify with the final means
    codes, errors = kmeans_classify( A, means, metric )

    # return the means, codes, and errors
    return (means, codes, errors)

# takes in the data, the number of clusters K, and an optional set of categories
# returns a numpy matrix with K rows, each representing a cluster mean.
def kmeans_init(d, K, categories=None):
    # If no categories are given, a simple way to select the means is to randomly
    # choose K data points (rows of the data matrix) to be the first K cluster means.
    if categories == None:
        randis = []
        i_range = range(len(d))
        for i in range(K):
            randi = random.choice(i_range)
            i_range.remove(randi)
            randis += [randi]

        meantrix = d[randis].copy()

    # If you are given an Nx1 matrix of categories/labels, then compute the mean values
    # of each category and return those as the initial set of means. You can assume the
    # categories are zero-indexed and range from 0 to K-1.
    else:
        meantrix = []
        rk = range(K)

        for i in rk:
            meantrix += [[]]

        # meantrix now contains indexes of each point in A, sorted by category
        for i in range(len(d)):
            meantrix[int(categories[0,i])] += [i]

        for i in rk:
            meantrix[i] = d[meantrix[i]].mean(0).tolist()[0]

        meantrix = np.matrix(meantrix)

    return meantrix

# takes in the data and cluster means
# returns a list of ID values and distances;
def kmeans_classify(d, means, metric):
    # The IDs should be the index of the closest cluster mean to the data point.
    # The default distance metric should be sum-2ed distance to the nearest cluster mean.

    distances = []
    ids = []

    for point in d:
        if metric == 'sum2ed':
            smallest_d = scipy.spatial.distance.euclidean(point, means[0])
        elif metric == 'corre':
            smallest_d = scipy.spatial.distance.correlation(point, means[0])

        ids += [0]
        distances += [smallest_d]

        for i,centroid in enumerate(means):
            if metric == 'sum2ed':
                potential_d = scipy.spatial.distance.euclidean(point, centroid)
            elif metric == 'corre':
                potential_d = scipy.spatial.distance.correlation(point, centroid)

            if potential_d < smallest_d:
                smallest_d = potential_d
                ids[-1] = i
                distances[-1] = smallest_d

    return np.matrix(ids).T, np.matrix(distances).T

if __name__ == "__main__":
    D = data.Data("clusterdata.csv")
    K = 2
    C = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
    print kmeans_init(D, K, C)