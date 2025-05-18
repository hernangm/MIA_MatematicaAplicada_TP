import helpers.jsonHelpers as jsonHelpers
import helpers.dataHelpers as dataHelpers
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
    enabledTests = [t for t in tests if t["enabled"]]
    for test in enabledTests:
        network = createNetwork(networkParams["layers"])
        algorithm, algorithmConfig = getAlgorithm(algorithms, test)
        scheduler, schedulerConfig = getScheduler(algorithms, test)
        params, prediction, trainLog, meta = runTest(test, data, network, algorithm, algorithmConfig, scheduler, schedulerConfig)
        testResults.append({
            "test": test,
            "params": params,
            "prediction": prediction,
            "trainLog": trainLog,
            "meta": meta
        })
    return testResults


if __name__ == '__main__':
    tests = jsonHelpers.load('hernan\\data\\tests.json')
    networkParams = jsonHelpers.load('hernan\\data\\network.json')
    data = dataHelpers.loadData('hernan\\data\\field.npy')
    algorithms = Algorithms({
        "sgd": sgd,
        "adam": adam,
        "rmsprop": rmsprop
    },{
        "trivial": schedule_trivial
    })
    testResults = runTests(algorithms, tests, networkParams, data[0])
    graph(data, testResults)