import numpy


def softmax(x: numpy.ndarray):
    return(numpy.exp(x - numpy.max(x)) / numpy.exp(x - numpy.max(x)).sum())