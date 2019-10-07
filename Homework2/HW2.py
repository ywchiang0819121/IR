from collections import Counter
import numpy as np
import math

doclist = []
quelist = []
with open('doc_list.txt', 'r') as docitr:
    doclist = docitr.read().splitlines()
with open('query_list.txt', 'r') as queitr:
    quelist = queitr.read().splitlines()

word = []
documentword = []
queryword = []
documentlennorm = []

for i in quelist:
    with open("Query/" + i) as f:
        s = f.read()
        brkdown = s.replace("-1","").split()
        word += brkdown
        queryword += [brkdown]
    
for i in doclist:
    with open("Document/" + i) as f:
        f.readline()
        f.readline()
        f.readline()
        s = f.read()
        brkdown = s.replace("-1","").split()
        documentword += [brkdown]
        documentlennorm.append(len(brkdown))

documentlennorm = np.array(documentlennorm)
documentlennorm = documentlennorm / np.mean(documentlennorm)
np.savetxt('norm.txt', documentlennorm)
word = list(set(word))

docuvec = []
quervec = []
idf = []

for i in range(len(quelist)):
    tmpvec = []
    for j in range(len(word)):
        t = queryword[i].count(word[j])
        if t > 0:
            tmpvec += [t / len(queryword[i])]
        else:
            tmpvec += [0]
    quervec += [(np.array(tmpvec))]

for i in range(len(doclist)):
    tmpvec = []
    for j in range(len(word)):
        t = documentword[i].count(word[j])
        if t > 0:
            tmpvec += [math.log10(1 + t / len(documentword[i]))]
        else:
            tmpvec += [0]
    docuvec += [(np.array(tmpvec))]

quervec = np.array(quervec)
docuvec = np.array(docuvec)
# np.savetxt("queryTF.txt", quervec)
# np.savetxt("documentTF.txt", docuvec)

docuidf = []

for i in range(len(word)):
    count = 0
    for j in range(len(documentword)):
        if word[i] in documentword[j]:
            count += 1
    try:
        docuidf += [np.array(np.log10(len(doclist) / (count+1)))]
    except:
        docuidf += [0]

docuidf = np.array(docuidf)

docuvec = np.array(docuvec) * np.array(docuidf)
quervec = np.array(quervec) * np.array(docuidf)

output = "Query,RetrievedDocuments\n"
for i in range(len(quelist)):
    output += quelist[i]+","
    scores = []
    for j in range(len(doclist)):
        scores += [np.dot(quervec[i],docuvec[j]) / (np.linalg.norm(quervec[i]) * np.linalg.norm(docuvec[j]))]
    n,fileName = zip(*sorted(zip(scores,doclist)))
    fileName = fileName[::-1]
    for j in range(len(fileName)):
        if j != len(fileName)-1:
            output += fileName[j]+" "
        else:
            output += fileName[j]+"\n"
      
with open("./result.txt","w") as f:
    f.write(output)