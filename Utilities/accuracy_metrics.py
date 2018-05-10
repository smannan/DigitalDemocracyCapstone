import numpy as np

# calculates recall if predicted transition 
# was correct within w, window size
# w = window size, int
# predicted = predicted transitions, list of ints
# actual = actual transitions, list of ints
def neighborhood_recall(predicted, actual, w):
	assert len(predicted) == len(actual)

	# number utterances
	n = len(predicted)

	# number of correctly predicted utterances
	correct = 0.0

	# indices of actual transitions
	indices = np.where(np.isin(actual, [1]))[0].tolist()

	for i in indices:
		start = i - w if i - w > 0 else 0
		stop = i + w if i + w < n else n-1
		window = predicted[start:stop+1]

		# at least one transition in the window
		if sum(window) > 0:
			correct += 1

	# number of "correctly" predicted / total
	return correct / (len(indices))

# for testing purposes
def main():
	w = 2

	# test case: contained in the window
	predicted = [0,1,0,0,0,0,0,0,0,1,0]
	actual = [0,1,0,0,0,0,0,1,0,0,0]
	print(neighborhood_recall(predicted, actual, w))

	# test case not contained in the window
	predicted = [0,1,0,0,0,0,0,0,0,0,1]
	actual = [0,1,0,0,0,0,0,1,0,0,0]
	print(neighborhood_recall(predicted, actual, w))

if __name__ == "__main__":
	main()