import sys
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import csv
from sklearn.linear_model import RidgeClassifier
import numpy as np

import matplotlib.pyplot as plt

categories = ['TRUE', 'FALSE']


maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

def loadData(listOfData):
    # file = open(fileName, 'r', encoding='utf8')
    fileName = ''
    file = None
    try:
        data = []
        for fileName in listOfData:
            file = open(fileName, 'r', encoding='utf8')
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
            file.close()
        dataTrain, dataTest = train_test_split(data, train_size=0.9, shuffle=True)
        corpusTrain = []
        valuesTrain = []
        corpusTest = []
        valuesTest = []
        safe = False
        count = 0
        removal = []
        for row in dataTrain:
            count += 1
            try:
                corpusTrain.append(row['text'])
                if row['label'].lower() == 'fake' or row['label'] == '1':
                    valuesTrain.append(0)
                    safe = True
                else:
                    valuesTrain.append(1)
            except AttributeError:
                removal.append(row)

        count = 0
        for row in dataTest:
            count += 1
            corpusTest.append(row['text'])
            if row['label'].lower() == 'fake' or row['label'] == '1':
                valuesTest.append(0)
            else:
                valuesTest.append(1)

        vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")
        y_train, y_test = valuesTrain, valuesTest
        x_train = vec.fit_transform(corpusTrain)
        x_test = vec.transform(corpusTest)

        featureNames = vec.get_feature_names_out()
        targetNames = categories
        if not safe:
            raise ValueError

    except ValueError:
        print(fileName + " dataset 'label' property improperly labeled")
        return None




    return x_train, x_test, y_train, y_test, featureNames, targetNames


def train(classifier, x_train, y_train, featureNames, topWordCount=5):
    classifier.fit(x_train, y_train)
    averageFeatureEffects = classifier.coef_ * np.asarray(x_train.mean(axis=0)).ravel()
    topTrue = np.argsort(averageFeatureEffects)[-topWordCount:]
    topFalse = np.argsort(averageFeatureEffects)[:topWordCount]
    topIndices = np.concatenate((topTrue, topFalse), axis=None)

    return topIndices


def test(classifier, x_test):
    return classifier.predict(x_test)


def plot(AFE, featureNames, barSize, padding, wordCount, fileName):
    topTrue = np.argsort(AFE)[-wordCount:]
    topFalse = np.argsort(AFE)[:wordCount]
    topIndices = np.concatenate((topTrue, topFalse), axis=None)
    topIndices = np.unique(topIndices)
    topWords = np.concatenate((featureNames[topTrue], featureNames[topFalse]))

    y_locs = np.arange(len(topIndices)) * (4 * barSize * padding)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y_locs, AFE[topIndices], align='center', height=barSize)
    ax.set_yticks(y_locs, labels=topWords)
    ax.set_xlabel("top 5 words for TRUE and FAKE articles for " + fileName)
    ax.set_title("Average feature effect on the original data")
    return fig, ax


def main():
    dataList = [
        "csvData/evaluationPreP.csv",
        "csvData/FAKE NEWS DATASET.csv",
        "csvData/fake_news.csv",
        "csvData/fake_news_dataset.csv",
        "csvData/fake_or_real_news.csv",
        "csvData/fake_or_real_news2PreP.csv",
    ]
    for i in range(len(dataList) + 1):

        topWordCount = 5
        name = ""
        print("Loading Data. . . ")
        if i < len(dataList):
            x_train, x_test, y_train, y_test, featureNames, targetNames = loadData([dataList[i]])
            name = dataList[i]
        else:
            x_train, x_test, y_train, y_test, featureNames, targetNames = loadData(dataList)
            name = "Super Database"
        print("Data Loaded")
        clf = RidgeClassifier(tol=0.01, solver='sparse_cg')
        print("Training on Data. . .")
        train(clf, x_train, y_train, featureNames, topWordCount=5)
        print("Training complete")
        print("Testing Classifier. . . ")
        pred = test(clf, x_test)
        print("Testing complete")

        barSize = 0.75
        padding = 0.3

        averageFeatureEffects = clf.coef_ * np.asarray(x_train.mean(axis=0)).ravel()
        plot(averageFeatureEffects, featureNames, barSize, padding, topWordCount, name)


        correctGuess = 0
        for j in range(len(pred)):
            if pred[j] == y_test[j]:
                correctGuess += 1
        print(f"correct guesses: {correctGuess}")
        print(f"Out of: {len(pred)}")
        print(f"{correctGuess / len(pred) * 100}%")

        if i >= len(dataList):
            with open('model.pkl', 'wb') as f:
                pickle.dump(clf, f)

        # plt.show()

# the code to load a pickle object, the model will be stored as model.pkl
# you then just run ckl.predict as above with a formatted database
# with open('model.pkl', 'wb') as f:
#     clf = pickle.load(f)


main()
