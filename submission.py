#!/usr/bin/python

import collections
from ctypes import windll
import random
from sys import exec_prefix
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    FeatureVector = collections.defaultdict(int)
    words = x.split(" ")
    for i in words: FeatureVector[i] += 1
    return FeatureVector
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent



def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    weights = collections.defaultdict(float)
    
    for i in range(numEpochs):
        for s in trainExamples:
            x = featureExtractor(s[0])
            y = s[1]
            grad1 = max(1 - dotProduct(weights, x) * y, 0)
            if grad1 > 0:
                for i in x:
                    x[i] = x.get(i) * -y * eta
                for i in x:
                    weights[i] -= x.get(i)
    #END
    return weights

trainExamples = (("hello world", 1), ("goodnight moon", -1))
testExamples = (("hello", 1), ("moon", -1))
learnPredictor(trainExamples, testExamples, extractWordFeatures, 20, 0.01)
############################################################
# Problem 3c: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.

    # Note that the weight vector can be arbitrary during testing. 
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this.
        phi = dict.fromkeys(weights, int)
        randomnum = 0
        phi1 = collections.defaultdict(int)
        for x in phi:
            while randomnum == 0:
                randomnum = random.randint(-1000,1000)
            phi1[x] = randomnum
        
        if dotProduct(weights, phi1) > 0: y = 1 
        else: y = -1
        #raise Exception("bruh let me test")
        # END_YOUR_CODE
        return phi1, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3e: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x_new = x.replace(" ", "")
        word_dict = collections.defaultdict(int)
        i = 0
        for i in range(len(x_new) -n + 1):
            word_dict[x_new[i: i + n]] += 1
        return word_dict
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

#testValuesOfN(3)
