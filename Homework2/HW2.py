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
        queryword += [brkdown]
    
for i in doclist:
    with open("Document/" + i) as f:
        f.readline()
        f.readline()
        f.readline()
        s = f.read()
        brkdown = s.replace("-1","").split()
        word += brkdown
        documentword += [brkdown]
        documentlennorm.append(len(brkdown))

documentlennorm = np.array(documentlennorm)
documentlennorm = documentlennorm / np.mean(documentlennorm)

queryword = np.array(queryword)
documentword = np.array(documentword)
docuidf = []

for i in range(len(quelist)):
    tmpvec = []
    for j in range(len(queryword[i])):
        t = 0
        for k in range(len(doclist)):
            if queryword[i][j] in documentword[k]:
                t += 1
        tmpvec += [np.array(np.log10((len(doclist) - t + 0.5) / (t + 0.5)))]
    docuidf += [(np.array(tmpvec))]

docuidf = np.array(docuidf)
k1 = 1.5
b = 1.0

output = "Query,RetrievedDocuments\n"
for i in range(len(quelist)):
    output += quelist[i]+","
    scores = []
    for j in range(len(doclist)):
        score = 0
        docCounter = Counter(documentword[j])
        queCounter = Counter(queryword[i])
        for k in range(len(queryword[i])):
            rawtf = docCounter[queryword[i][k]]
            score += docuidf[i][k] * (rawtf * (k1 + 1) / (rawtf + k1 * (1 - b + b * documentlennorm[j])))
        scores += [score]
    n,fileName = zip(*sorted(zip(scores,doclist)))
    fileName = fileName[::-1]
    for j in range(len(fileName)):
        if j != len(fileName)-1:
            output += fileName[j]+" "
        else:
            output += fileName[j]+"\n"
      
with open("./result.txt","w") as f:
    f.write(output)