import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
import seaborn as sn
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm, multiclass
import warnings
import time
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# plot c vs gamma for each degree given the polynomial kernel grid search results
def plotGS3D(results, param1, param2, param3, foldNumber, classification):

    # list of figures and polynomial degrees
    figs = []
    degrees = [2, 3, 4]

    # Change classification to actual words
    if classification == 'ovo':
        cl = "One Vs. One"
    else:
        cl = "One Vs. All"

    param1Name = "C"
    param2Name = "Gamma"

    # get the mean score of the cv's and reshape for graphing purposes
    meanScore = results['mean_test_score']
    meanScore = np.array(meanScore).reshape(len(param3), len(param2), len(param1))

    # plot c vs gamma for each degree
    for i,matrix in enumerate(meanScore[:]):
        fig, axes = plt.subplots(1,1)

        for index, value in enumerate(param2):
            axes.plot(param1, matrix[index, :], '-o', label=param2Name + ': ' + str(value), alpha=0.7)

        axes.set_title("Fold #" + str(foldNumber) + " GridSearch Score (Degree " + str(degrees[i]) +" Poly " + cl + ")", fontsize=16, fontweight= 'bold')
        axes.set_xlabel(param1Name, fontsize=12)
        axes.set_ylabel('CV Average Score', fontsize=12)
        axes.legend(loc='best', fontsize=11)
        axes.grid('on')

        figs.append(fig)

    return figs

# plot the confusion matrices
def plotConfusionMatrix(confusion, kernel, classification):

    # Change classification to actual words
    if classification == 'ovo':
        classification = "One Vs. One"
    else:
        classification = "One Vs. All"

    # plot the matrix as heatmap
    fig = plt.subplot()
    sn.heatmap(confusion, ax=fig, fmt ='g', cmap='Oranges', annot=True)
    plt.savefig('Confusion Matrix - ' + kernel.title() + " " + classification + ".png")

    plt.close()

    return

# plot the time versus the accuracy given the ovo and ovr results
def plotTimeVAccuracy(kernel, ovoTime, ovrTime, ovoAcc, ovrAcc):

    fig, axes = plt.subplots(1,1)
    axes.bar(ovoTime, ovoAcc, width=0.5, label= "One Vs. One")
    axes.text(ovoTime, ovoAcc+0.5, str(round(ovoAcc,2)), fontweight='bold', color='blue')
    axes.bar(ovrTime, ovrAcc, width=0.5, label= "One Vs. All")
    axes.text(ovrTime, ovrAcc+0.5, str(round(ovrAcc,2)), fontweight='bold', color='orange')

    axes.set_title(kernel.title() + " Kernel: Time Vs. Accuracy", fontsize=20, fontweight='bold')
    axes.set_xlabel("Time (in seconds)", fontsize=16)
    axes.set_ylabel('Mean Accuracy (%)', fontsize=16)
    axes.legend(loc="best", fontsize=15)
    axes.grid('on')

    return fig


# plot the results of the grid search
def plotGridSearch(results, param1, param2, param1Name, param2Name, foldNumber, kernel, classification):

    # change classification for labeling purposes
    if classification == 'ovo':
        cl = 'One Vs One'
    else:
        cl = 'One Vs All'

    # ensure the kernel title is capitalized
    kernel = kernel.title()

    # get the mean score of the cv's and reshape for graphing purposes
    meanScore = results['mean_test_score']
    meanScore = np.array(meanScore).reshape(len(param2), len(param1))

    fig, axes = plt.subplots(1,1)

    for index, value in enumerate(param2):
        axes.plot(param1, meanScore[index, :], '-o', label=param2Name + ': ' + str(value), alpha=0.7)

    axes.set_title("Fold #" + str(foldNumber) + " GridSearch Score (" + kernel + " " + cl + ")", fontsize=16, fontweight= 'bold')
    axes.set_xlabel(param1Name, fontsize=12)
    axes.set_ylabel('CV Average Score', fontsize=12)
    axes.legend(loc='best', fontsize=11)
    axes.grid('on')

    return fig

# perform svm with the provided kernel and type of classification
def svm(data, kernel, classification, weighted=False, plot=False, onePlot=False):

    # list of labels
    labels = ["1", "2", "3", "5", "6", "7"]

    # initialize lists for confusion matrices, accuracies, and best parameters
    confusions = []
    accuracies = []
    bestParams = []
    plots = []
    timePerFold = []

    # set the proper classification
    if classification == 'ovo' :
        # if weighted option is chosen, balance the dataset so that the weights are inversely proportional to frequency
        if weighted == False:
            svc = multiclass.OneVsOneClassifier(SVC(kernel=kernel))
        else:
            svc = multiclass.OneVsOneClassifier(SVC(kernel=kernel, class_weight='balanced'))

        # initialize parameters for GridSearchCV
        params = {
            "estimator__C": [0.01, 1, 10, 100, 500, 1000],
            'estimator__gamma': [0.01, 1, 10]
        }

        c = "estimator__C"
        gamma = 'estimator__gamma'
        degree = 'estimator__degree'
    elif classification == 'ovr':
        svc = SVC(kernel=kernel)

        # initialize parameters for GridSearchCV
        params = {
            "C": [0.01, 1, 10, 100, 500, 1000],
            "gamma": [0.01, 1, 10]
        }

        c = "C"
        gamma = 'gamma'
        degree = 'degree'
    else:
        print('Invalid Classifier Type')
        return

    # if we have a polynomial kernel we want to reduce the penalty parameter to reduce training time
    if kernel == 'poly':
        params[c] = [0.001, 0.01, 0.1]
        params[degree] = [2,3,4]
    elif kernel == 'sigmoid':
        params[gamma].append(0.001)
        params[gamma].append(0.0001)

    # split the data into features and labels
    data = data.values
    x = data[:,0:9]
    y = data[:,9:]

    # normalize the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # perform k-fold cross validation, in this case we're using 5 folds
    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    foldNumber = 0
    totalTime = 0
    for train, test in kf.split(data):

        # increase which fold we're on
        foldNumber += 1

        # get training and test splits
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        y_train, y_test = y_train.ravel(), y_test.ravel()

        # perform Grid Search on the training sets
        clf = GridSearchCV(svc, cv=5, param_grid=params, iid=True)

        # start time for the training
        t0 = time.time()

        # fit the model
        clf.fit(x_train, y_train)

        # end time for the training
        t1 = time.time()
        timePerFold.append(t1-t0)
        totalTime += t1-t0

        # test the model
        y_pred = clf.predict(x_test)

        # save the best parameters for this fold
        bestParams.append(clf.best_params_)

        # plot the grid search results, save for later
        if (plot and kernel != 'poly') or onePlot:
            plots.append(plotGridSearch(clf.cv_results_, params[c], params[gamma], "C", "Gamma", foldNumber, kernel, classification))
        elif plot and kernel == 'poly':
            plots.append(plotGS3D(clf.cv_results_, params[c], params[gamma], params[degree], foldNumber, classification))

        # append confusion matrix and accuracy to respective list
        accuracies.append(accuracy_score(y_test, y_pred))
        confusions.append(confusion_matrix(y_test, y_pred))

    # get mean accuracy
    meanAccuracy = np.mean(accuracies) * 100

    # add weighted to kernel name if applicable
    if weighted:
        kernel += "_Weighted"

    # save all the plots as a pdf
    if plot:

        # if we are on the polynomial kernel, flatten the list of lists
        if kernel == 'poly' or kernel == 'poly_Weighted':
            plots = [plot for subplot in plots for plot in subplot]

        file = pdf.PdfPages(kernel.title() + "_Kernel_" + classification.upper() + "_Classification.pdf")
        for fig in plots:
            file.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        file.close()

        # get averaged confusion matrix
        confusions = [pd.DataFrame(data=c, columns=labels, index=labels) for c in confusions]
        concatCM = pd.concat(confusions)
        cm_total = concatCM.groupby(concatCM.index)
        cm_average = cm_total.mean()

        # plot average confusion matrix
        plotConfusionMatrix(cm_average, kernel, classification)
    elif onePlot:
        plots[1].show()


    # print some useful information
    print("-"*300)
    print("Classification: ", classification)
    print("Kernel: ", kernel)
    print("Mean Accuracy: ", meanAccuracy)
    print("Time it took to train: ", totalTime)
    print("Time per Fold", timePerFold)
    print("Best Parameters per Fold: ", bestParams)
    print()
    return totalTime, meanAccuracy

# open the data file and run SVM
def main():

    # input = "../Data/glass.data"
    input = "glass.data"
    headers = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "type"]
    data = pd.read_csv(input, names=headers)
    data.drop(["Id"], axis=1, inplace=True)

    # if we'd like to plot all the results, set plot to true
    plot = False
    onePlot = True

    # list for each type of kernel
    kernels = ['linear', 'rbf', 'sigmoid', 'poly']

    if plot:
        onePlot = False
        timeVacc = pdf.PdfPages("Time_VS_Accuracy.pdf")

    # run each type of kernel with both 1v1 and 1vAll
    for k in kernels:
        ovoTime, ovoAccuracy = svm(data, kernel=k, classification='ovo', plot=plot, onePlot=onePlot)
        ovrTime, ovrAccuracy = svm(data, kernel=k, classification='ovr', plot=plot)
        onePlot = False
        print("*"*300)

        if plot:
            # plot the time vs the accuracy and save to pdf
            fig = plotTimeVAccuracy(k, ovoTime, ovrTime, ovoAccuracy, ovrAccuracy)
            timeVacc.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    if plot:
        timeVacc.close()

    # run each type of kernel with 1v1 where the classes are reweighted
    for k in kernels:
        ovoTime, ovoAccuracy = svm(data, kernel=k, classification='ovo', weighted=True, plot=plot)


if __name__ == "__main__":
    main()