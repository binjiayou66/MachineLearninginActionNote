from math import log
import operator

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVce in dataSet:
		currentLabel = featVce[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [item[i] for item in dataSet]
		uniqueFeats = set(featList)
		newEntropy = 0.0
		for feat in uniqueFeats:
			subDataSet = splitDataSet(dataSet, i, feat)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys(): 
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def createTree(dataSet, labels):
	classList = [item[-1] for item in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	tree = {bestFeatLabel : {}}
	del(labels[bestFeat])
	uniqueFeats = set([item[bestFeat] for item in dataSet])
	for feat in uniqueFeats:
		subLabels = labels[:]
		tree[bestFeatLabel][feat] = createTree(splitDataSet(dataSet, bestFeat, feat), subLabels)
	return tree

def createDataSet():
	dataSet = [
		[1, 1, 'yes'],
		[1, 1, 'yes'],
		[1, 0, 'no'],
		[0, 1, 'no'],
		[0, 1, 'no']
	]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

data, labels = createDataSet()

# shannonEnt = calcShannonEnt(data)
# print(shannonEnt)

# bestFeat = chooseBestFeatureToSplit(data)
# print(bestFeat)

tree = createTree(data, labels)
print(tree)






