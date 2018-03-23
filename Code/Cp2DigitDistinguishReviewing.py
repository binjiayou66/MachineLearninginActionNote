from numpy import *
import os
import operator

'''
k-近似算法，归一处理，训练样本 // 处理可量化指标分类问题
'''

def createDataSet():
	group = array(([1.1, 1.1], [1.1, 0.9], [0.1, 0.1], [0.1, 0.1]))
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def kNear(inX, group, labels, top):
	shapeX = group.shape[0]
	matrix = tile(inX, (shapeX, 1))
	diffMatrix = matrix - group
	sqDiffMatrix = diffMatrix ** 2
	sumSqDiffMatrix = sqDiffMatrix.sum(axis=1)
	distances = sumSqDiffMatrix ** 0.5
	print(sort(distances))
	indexs = distances.argsort()
	nearCount = {}
	for i in range(top):
		nearCount[labels[indexs[i]]] = nearCount.get(labels[indexs[i]], 0) + 1
	nears = sorted(nearCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return nears[0][0]

def imgToVector(filename):
	result = zeros((1, 1024))
	handle = open(filename)
	for i in range(32):
		line = handle.readline()
		for j in range(32):
			result[0,32*i+j] = int(line[j])
	return result

def digitDistinguish():
	labels = []
	trainingFileList = os.listdir('/Users/andy/Desktop/PythonCode/Resources/trainingDigits')
	trainingNum = len(trainingFileList)
	trainingMatrix = zeros((trainingNum, 1024))
	for i in range(trainingNum):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		numStr = int(fileStr.split('_')[0])
		labels.append(numStr)
		trainingMatrix[i,:] = imgToVector('/Users/andy/Desktop/PythonCode/Resources/trainingDigits/%s' %fileNameStr)
	errorCount = 0.0
	testFileList = os.listdir('/Users/andy/Desktop/PythonCode/Resources/testDigits')
	testNum = len(testFileList)
	for i in range(testNum):
		testMtrix = imgToVector('/Users/andy/Desktop/PythonCode/Resources/testDigits/%s' % testFileList[i])
		result = kNear(testMtrix, trainingMatrix, labels, 3)
		real = int(testFileList[i].split('_')[0])
		print('real = %s, result = %s' % (real, result))
		if real != result: 
			print("not same = %s, %s, %s" % (real, result, testFileList[i]))
			errorCount += 1.0
	print('total error count: %d, total error rate: %f' % (errorCount, errorCount / float(testNum)))

def test():
	group, labels = createDataSet()
	print(kNear([0.1, 0.3], group, labels, 3))

digitDistinguish()



