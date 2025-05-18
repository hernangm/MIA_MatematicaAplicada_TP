import helpers.json as json
import dataLoader as dataLoader
from algorithms import Algorithms
from algorithms.optimizers.sgd import sgd
from algorithms.optimizers.adam import adam
from algorithms.optimizers.rmsprop import rmsprop
from algorithms.schedulers.trivial import schedule_trivial
from testRunner import runTest
from neuralNetwork import createNetwork
from graphs import graph

def getAlgorithm(algorithms, test):
    c = test["config"]["algorithm"]
    algorithm = algorithms.getAlgorithm(c["name"])
    return (algorithm, c["config"])

def getScheduler(algorithms, test):
    c = test["config"]["scheduler"]
    scheduler = algorithms.getScheduler(c["name"])
    return (scheduler, c["config"])


def runTests(algorithms, tests, networkParams, data):
    testResults = []
    for test in tests:
        network = createNetwork(networkParams["layers"])
        algorithm, algorithmConfig = getAlgorithm(algorithms, test)
        scheduler, schedulerConfig = getScheduler(algorithms, test)
        results, meta = runTest(test, data, network, algorithm, algorithmConfig, scheduler, schedulerConfig)
        testResults.append((test, results, meta))
    return testResults


if __name__ == '__main__':
    tests = json.load('hernan\\config\\tests.json')
    networkParams = json.load('hernan\\config\\network.json')
    data = dataLoader.loadData('hernan\\data\\field.npy')
    algorithms = Algorithms({
        "sgd": sgd,
        "adam": adam,
        "rmsprop": rmsprop
    },{
        "trivial": schedule_trivial
    })
    testResults = runTests(algorithms, tests, networkParams, data)
    graph(testResults)