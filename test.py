
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import csv
from sklearn.linear_model import RidgeClassifier
import numpy as np
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

categories = ['TRUE', 'FALSE']

def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs)


def loadData(verbose=False, remove=()):
    file = open('csvData/fake_or_real_news.csv', 'r', encoding='utf8')
    reader = csv.DictReader(file)
    data = []
    for row in reader:
        data.append(row)
    dataTrain, dataTest = train_test_split(data, train_size=0.9, shuffle=True)
    corpusTrain = []
    valuesTrain = []
    corpusTest = []
    valuesTest = []
    for row in dataTrain:
        corpusTrain.append(row['text'])
        if row['label'] == 'FAKE' or row['label'] == '1':
            valuesTrain.append(0)
        else:
            valuesTrain.append(1)
    for row in dataTest:
        corpusTest.append(row['text'])
        if row['label'] == 'FAKE' or row['label'] == '1':
            valuesTest.append(0)
        else:
            valuesTest.append(1)

    vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")
    y_train, y_test = valuesTrain, valuesTest
    x_train = vec.fit_transform(corpusTrain)
    x_test = vec.transform(corpusTest)

    featureNames = vec.get_feature_names_out()
    targetNames = categories

    return x_train, x_test, y_train, y_test, featureNames, targetNames


def main():
    print("Loading Data")
    x_train, x_test, y_train, y_test, featureNames, targetNames = loadData()
    print("Data Loaded")
    clf = RidgeClassifier(tol=0.01, solver='sparse_cg')
    print("Fitting Data")
    clf.fit(x_train, y_train)
    print("Data Fitted")
    pred = clf.predict(x_test)

    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    ax.xaxis.set_ticklabels(targetNames)
    ax.yaxis.set_ticklabels(targetNames)
    ax.set_title(
        f"Confusion Matrix for {clf.__class__.__name__}\non the original documents"
    )
    plt.show()


    # plotting features
    averageFeatureEffects = clf.coef_ * np.asarray(x_train.mean(axis=0)).ravel()

    top5True = np.argsort(averageFeatureEffects)[-5:]
    top5False = np.argsort(averageFeatureEffects)[:5]
    topIndices = np.concatenate((top5True, top5False), axis=None)
    # top = pd.DataFrame(featureNames[top5True], columns=["True"])
    # top["False"] = featureNames[top5False]
    topWords = np.concatenate((featureNames[top5True], featureNames[top5False]))
    # for i, label in enumerate(targetNames):
    #     top5 = np.argsort(averageFeatureEffects)[-5:]
    #     if i == 0:
    #         top = pd.DataFrame(featureNames[top5], columns=[label])
    #         topIndices = top5
    #     else:
    #         top[label] = featureNames[top5]
    #         topIndices = np.concatenate((topIndices, top5), axis=None)
    topIndices = np.unique(topIndices)
    topWords = np.concatenate((featureNames[top5True], featureNames[top5False]))
    predictiveWords = featureNames[topIndices]

    barSize = 0.75
    padding = 0.3
    spacing = 0.1
    error = 0.1
    y_locs = np.arange(len(topIndices)) * (4 * barSize * padding)

    fig, ax = plt.subplots()

    yPos = np.arange(len(topIndices))

    ax.barh(y_locs, averageFeatureEffects[topIndices], align='center', height=barSize)
    ax.set_yticks(y_locs, labels=topWords)
    ax.set_xlabel("top 5 words for TRUE and FAKE articles")
    ax.set_title("Average feature effect on the original data")

    correctGuess = 0
    for i in range(len(pred)):
        if pred[i] == y_test[i]:
            correctGuess += 1
    print(f"correct guesses: {correctGuess}")
    print(f"Out of: {len(pred)}")
    print(f"{correctGuess / len(pred) * 100}%")

    plt.show()


if __name__ == '__main__':
    main()