import re

def preprocess1():
    file = open("csvData/evaluation.csv", "r", encoding="utf8")
    content = ""
    line = file.readline()
    count = 0
    while line:
        content += line
        line = file.readline()
        print(count)
        count += 1
    content = content.replace(",", " ")
    content = content.replace(";", ",")
    file.close()
    with open("csvData/evaluationPreP.csv", "w", encoding="utf8") as text_file:
        text_file.write(content)


# def preprocess2():
#     file = open("csvData/WELFake_Dataset.csv", "r", encoding="utf8")
#     try:
#
#         contentList = file.read().splitlines(True)
#         content = ''.join(contentList)
#         done = False
#         while not done:
#             previous = content
#             content = content.replace("\n\n", "\n")
#             if content == previous:
#                 done = True
#         contentSplit = content.split("\n")
#         for i in range(len(contentSplit)):
#             split = contentSplit[i].split(",")
#             if split[-1] == '0':
#                 split[-1] = '1'
#             elif split[-1] == '1':
#                 split[-1] = '0'
#             if i % 1000 == 0:
#                 print(i)
#             for j in range(len(split)):
#                 if j != 0:
#                     split[j] = ',' + split[j]
#                 if j == len(split) - 1:
#                     split[j] = split[j] + '\n'
#             contentSplit[i] = ''.join(split)
#         file.close()
#         content = ''.join(contentSplit)
#         with open("csvData/WELFake_DatasetPreP.csv", "w", encoding="utf8") as text_file:
#             text_file.write(content)
#     except:
#         file.close()
#         print("ended run")

def preprocess3():
    file = open("csvData/True.csv", "r", encoding="utf8")

    contentList = file.read().splitlines(True)
    for i in range(len(contentList[1:])):
        contentList[i + 1] = contentList[i + 1][:-1] + ',0\n'
    content = ''.join(contentList)

    file.close()

    file = open("csvData/Fake.csv", "r", encoding="utf8")

    contentList = file.read().splitlines(True)
    for i in range(len(contentList[1:])):
        contentList[i + 1] = contentList[i + 1][:-1] + ',1\n'
    content += ''.join(contentList[1:])

    with open("csvData/TrueFakePreP.csv", "w", encoding="utf8") as text_file:
        text_file.write(content)



    file.close()
    print("ended run")



def preprocess4():
    file = open("csvData/fake_or_real_news2.csv", "r", encoding="utf8")
    content = ''
    contentList = file.read().splitlines(True)

    for i in range(len(contentList[1:])):
        realData = [None, None]
        split = contentList[i + 1].split(",")
        realData[1] = "," + split[-1]
        realData[0] = ' '.join(split[:-1])
        contentList[i + 1] = ''.join(realData)

    content += ''.join(contentList[1:])

    with open("csvData/fake_or_real_news2PreP.csv", "w", encoding="utf8") as text_file:
        text_file.write(content)


def preprocess5():
    file = open("csvData/TrueFakePreP.csv", "r", encoding="utf8")
    content = ''
    contentList = file.read().splitlines(True)
    for i in range(len(contentList[1:])):
        contentList[i + 1] = contentList[i + 1].replace("image", "")
        contentList[i + 1] = contentList[i + 1].replace("Image", "")
        contentList[i + 1] = contentList[i + 1].replace("featured", "")
        contentList[i + 1] = contentList[i + 1].replace("Featured", "")
    content += ''.join(contentList)


    with open("csvData/TrueFakePrePreP.csv", "w", encoding="utf8") as text_file:
        text_file.write(content)




preprocess5()
