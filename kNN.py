import os
import os.path
from matplotlib.image import imread
import numpy as np
import operator
from collections import Counter


def getImages(path):

	images = []

	for dirpath, dirnames, filenames in os.walk(path):
		for filename in [f for f in filenames if f.endswith(".JPEG")]:
			images.append(os.path.join(dirpath, filename))

	return images

def imagesAsArrays(imagelist):
	# converts to array
	imageArray = [imread(i) for i in imagelist]
	return imageArray

def getTrainLabels(list):
	# gets the category
	labels = [os.path.basename(i)[0:9] for i in list]
	return labels

def getTestLabels(file):
	labels = []
	z = open(file, 'r')
	for line in z.readlines():
		cols = line.split('\t')
		labels.append(cols[1])
	z.close()
	return labels

def KNNPreds(trainImages, trainLabels, testImages, k):

	preds = []

	for i in range(len(testImages)):

		# print(i)

		im = testImages[i]

		distances = []

		for j in range(len(trainImages)):

			d = absoluteDistance(im, trainImages[j])

			distances.append((j, d))

		distances.sort(key=operator.itemgetter(1))

		neighbors = []

		for x in range(k):

			neighbors.append(trainLabels[distances[x][0]])

		l = getVotes(neighbors)

		preds.append(l)

	return preds


def KNNPredsTop5(trainImages, trainLabels, testImages, k):

	preds = []

	for i in range(len(testImages)):

		# print(i)

		im = testImages[i]

		distances = []

		for j in range(len(trainImages)):

			d = absoluteDistance(im, trainImages[j])

			distances.append((j, d))

		distances.sort(key=operator.itemgetter(1))

		neighbors = []

		for x in range(k):

			neighbors.append(trainLabels[distances[x][0]])

		l = getVotesTop5(neighbors)

		preds.append(l)

	return preds


def getVotes(list):

	c = Counter(list)
	v = c.most_common()[0][0]
	if v == 1:
		return list[0]
	else:
		return c.most_common()[0][0]

def getVotesTop5(list):

	c = Counter(list)
	v = c.most_common()[0][0]
	if v == 1:
		return list[0]
	else:
		return [c.most_common()[0][0], c.most_common()[1][0], c.most_common()[2][0], c.most_common()[3][0], c.most_common()[4][0]]

def absoluteDistance(a1, a2):

	return np.linalg.norm(a1.flatten()-a2.flatten())


if __name__ == "__main__":

	# get the train images
	trainImageList = getImages("./tiny-imagenet-200/train/")
	# print(len(trainImageList))

	# get the array of images as arrays 
	trainImageArray = imagesAsArrays(trainImageList)
	# print(len(trainImageArray))

	trainLabels = getTrainLabels(trainImageList)
	# print(trainImageList[1102], trainLabels[1102])

	# see why some images are gray scale
	# for i in range(len(trainImageArray)):
	# 	if trainImageArray[i].shape != (64,64,3):
	# 		print(trainImageList[i], trainLabels[i])

	# fix gray scale images - make them 3D
	for i in range(len(trainImageArray)):
		if trainImageArray[i].shape != (64,64,3):
			trainImageArray[i] = np.stack((trainImageArray[i], trainImageArray[i], trainImageArray[i]), axis = -1)
			# print(trainImageArray[i].shape)

	# get the test images
	testImageList = getImages("./tiny-imagenet-200/val/")
	# print(len(testImageList))

	# get the array of images as arrays
	testImageArray = imagesAsArrays(testImageList)
	# print(len(testImageArray))

	# fix gray scale images - make them 3D
	for i in range(len(testImageArray)):
		if testImageArray[i].shape != (64,64,3):
			testImageArray[i] = np.stack((testImageArray[i], testImageArray[i], testImageArray[i]), axis = -1)
			# print(testImageArray[i].shape)

	testLabels = getTestLabels("./tiny-imagenet-200/val/val_annotations.txt")
	# print(len(testLabels))
	# print(testLabels[0])


	##### TESTING ######

	# make sample smaller
	sampleVal = 100

	# testing for correctness
	#trainImageArray.extend(testImageArray[0:sampleVal])
	#trainLabels.extend(testLabels[0:sampleVal])

	k = 50
	#######

	### Running KNN - top 1
	# predictions = KNNPreds(trainImageArray, trainLabels, testImageArray[0:sampleVal], k)
	#print(predictions)
	#print(testLabels[0:sampleVal])

	# correct = 0
	# for i in range(len(predictions)):
	# 	if testLabels[i] == predictions[i]:
	# 		correct += 1

	# print(correct, len(testLabels[0:sampleVal]))


	### Running KNN - top 5
	predictions5 = KNNPredsTop5(trainImageArray, trainLabels, testImageArray[0:sampleVal], k)
	#print(predictions5)
	#print(testLabels[0:sampleVal])

	correct = 0
	for i in range(len(predictions5)):
		if testLabels[i] in predictions5[i]:
			correct += 1

	print(correct, len(testLabels[0:sampleVal]))
