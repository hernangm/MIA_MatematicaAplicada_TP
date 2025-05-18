from datetime import datetime
import time
from optimizer import optimize

def formatTime(time):
    return datetime.fromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S")

def runTest(test, data, network, algorithm, algorithmConfig, scheduler, schedulerConfig):
    testName = test["name"]
    startTime= time.time()
    print(f"{formatTime(startTime)} Test {testName} started...")
    results = optimize(network, data, test["config"], algorithm, algorithmConfig, scheduler, schedulerConfig)
    endTime = time.time()
    duration = endTime - startTime
    print(f"{formatTime(endTime)} Test {testName} finished in {round(duration, 4)}.")
    meta = {
        "startTime": startTime,
        "endTime": endTime,
        "duration": duration
    }
    return (results, meta)