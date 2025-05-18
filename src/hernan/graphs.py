import matplotlib.pyplot as plt

def graph(testResults):
    plt.figure()
    for tr in testResults:
        plt.semilogy([tt["trainLoss"] for tt in tr[1]], label= tr[0]["name"])
    plt.legend()
    plt.show()