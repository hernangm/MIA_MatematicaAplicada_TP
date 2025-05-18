import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def graphData(axis, data, size):
    axis.imshow(data.reshape(size).T, origin='lower', cmap='jet')

def graphError(axis, trainLog, test, graphOptions):
    axis.semilogy([tt["trainLoss"] for tt in trainLog], label= test["name"], color=graphOptions["color"])
    # axis.xlabel("Epochs")

def graph(data, testResults):
    sizes = data[1]

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2 + len(testResults), 2)  # 2 rows, 2 columns

    i = 0
    ax1 = fig.add_subplot(gs[0, 0])
    graphData(ax1, data[0][1], sizes)

    for tr in testResults:
        i += 1
        test = tr["test"]
        graphOptions = test["graph"]
        if (graphOptions["enabled"]):
            axisLeft = fig.add_subplot(gs[i, 0])  # row 1, column 0
            axisRight = fig.add_subplot(gs[i, 1])  # row 1, column 1
            graphData(axisLeft, tr["prediction"], sizes)
            graphError(axisRight, tr["trainLog"], test, graphOptions)

    i += 1
    bottomAxis = fig.add_subplot(gs[i, :])
    for tr in testResults:
        test = tr["test"]
        graphOptions = test["graph"]
        if (graphOptions["enabled"]):
            graphError(bottomAxis, tr["trainLog"], test, graphOptions)
    
    plt.legend()
    plt.show(block=True)