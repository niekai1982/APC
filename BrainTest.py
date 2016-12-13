import os
import cv2
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, TanhLayer, SoftmaxLayer
from pybrain.utilities import percentError
from time import time
from common import mosaic
from get_feature import get_hog_feature, get_icf_feature
from get_feature import EXT_DICT
import feature.stub as stub


# data = datasets.load_iris()
TEST_MODEL = 1
TRAIN_MODEL = 0


def Generate_dataset(num_samples):
    # Generate a dataset and plot it
    np.random.seed(0)
    X, y = datasets.make_moons(num_samples, noise=0.10)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
    return X, y


def Get_data(sample_path):
    data = np.load(sample_path)['data']
    target = np.load(sample_path)['target']
    return data, target


def fast_fnn(num_features, num_classes, num_ep):
    fnn = buildNetwork(num_features, num_ep, num_classes,
                       hiddenclass=TanhLayer, outclass=SoftmaxLayer)
    print fnn['in']
    print fnn['hidden0']
    print fnn['out']
    return fnn


def build_fnn():
    fnn = FeedForwardNetwork()
    inLayer = LinearLayer(2)
    hiddenLayer = TanhLayer(50)
    outLayer = SoftmaxLayer(2)
    fnn.addInputModule(inLayer)
    fnn.addModule(hiddenLayer)
    fnn.addOutputModule(outLayer)
    return fnn


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    return probs, np.argmax(probs, axis=0)


def load_model(path, input_dim, hidden_dim, out_dim):
    params = np.load(path)
    b2_dim = out_dim
    b1_dim = hidden_dim
    W1_dim = (hidden_dim, input_dim)
    W2_dim = (out_dim, hidden_dim)

    b2 = params[:b2_dim]
    b1 = params[b2_dim:(b1_dim + b2_dim)]
    W1 = params[(b1_dim + b2_dim):(b1_dim + b2_dim + W1_dim[0] * W1_dim[1])]
    W2 = params[(b1_dim + b2_dim + W1_dim[0] * W1_dim[1]):]

    W1.shape = W1_dim
    W2.shape = W2_dim

    model = {'W1': W1.T, 'b1': b1, 'W2': W2.T, 'b2': b2}
    return model


def vis_res(fnn):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    griddata = ClassificationDataSet(2, 1, nb_classes=2)
    for i in xrange(xx.size):
        griddata.addSample([xx.ravel()[i], yy.ravel()[i]], [0])
    # this is still needed to make the fnn feel comfy
    griddata._convertToOneOfMany()
    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1).reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, out, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def norm_data(X, y):
    num_features = X.shape[-1]
    num_classes = len(set(y))
    data = ClassificationDataSet(num_features, 1, nb_classes=num_classes)
    y.shape = -1, 1
    data.setField('input', X)
    data.setField('target', y)
    data._convertToOneOfMany()
    return data


if __name__ == '__main__':

    # NUM_FEATURES = 2
    # NUM_CLASSES  = 2
    # NUM_SAMPLES  = 200
    # X, y = Generate_dataset(NUM_SAMPLES)

    model_path = r'E:\APC\sample_test\model.npy'

    num_hidden_ep = 200

    if TRAIN_MODEL:

        sample_path = r'E:\APC\sample_test\data_test.npz'

        X, y = Get_data(sample_path)

        num_features = X.shape[-1]
        num_classes = len(set(y))

        fnn = fast_fnn(num_features, num_classes, num_hidden_ep)
        data = norm_data(X, y)

        tsd, trd = data.splitWithProportion(0.25)
        print len(trd)
        #trainer = BackpropTrainer(fnn, trd, verbose=True)

        trainer = BackpropTrainer(fnn, trd, verbose=True, weightdecay=0.01, learningrate=0.001)

        trainer.trainUntilConvergence(maxEpochs=100)
        tstresult = percentError(trainer.testOnClassData(dataset=tsd),
                                 tsd['class'])
        print "epoch: %4d" % trainer.totalepochs, \
              "  test error: %5.2f%%" % tstresult

        np.save(model_path, fnn.params)

    if TEST_MODEL:

        test_path = r'E:\APC\sample_test\0'
        vector_stub_path = r'e:\APC\feature\canditate.vec'
        vec_stu = stub.read(vector_stub_path)
        test_files = os.listdir(test_path)

        # num_features = len(
        #     get_hog_feature(cv2.imread(os.path.join(test_path, test_files[0]))))
        num_features = len(vec_stu)
        num_classes = 2

        model = load_model(model_path, input_dim=num_features,
                           hidden_dim=num_hidden_ep, out_dim=num_classes)

        head_p = []
        head_n = []

        for elem in test_files:
            if os.path.splitext(elem)[-1] not in EXT_DICT:
                continue
            data = cv2.imread(os.path.join(test_path, elem))
            start = time()
            feature = get_icf_feature(data, vec_stu)
            prob, res = predict(model, feature)
            print "spend time: %f" % (time() - start)
            if res == 1:
                head_n.append(cv2.resize(data, (32, 32)))
            else:
                head_p.append(cv2.resize(data, (32, 32)))

        plt.subplot(1,2,1)
        plt.imshow(mosaic(20, head_p))#, aspect='auto')
        plt.title('HEAD')
        plt.subplot(1,2,2)
        plt.imshow(mosaic(20, head_n))
        plt.title('NO HEAD')
        plt.show()
