import numpy as np

def sigmoid(X):
	return 1 / (1 + np.exp(-X))

#Change this function
def softmax(X):
	Z = np.sum(np.exp(X), axis=1)
	Z = Z.reshape(Z.shape[0], 1)
	return np.exp(X) / Z

def truthTable_neural(step):
	
	#Make features and labels optional
	features = np.matrix('1.0, 1.0;'
						 '1.0, 0.0;'
						 '0.0, 1.0;'
						 '0.0, 0.0')
						 
	labels = np.matrix('1.0, 0.0;'
					   '0.0, 1.0;'
					   '0.0, 1.0;'
					   '1.0, 0.0')

	#Initial random weights
	weightsLayer1 = np.random.rand(2,4)
	weightsLayer2 = np.random.rand(4,2)
	
	#Gradient weights and biases
	_0_W = np.zeros([2,4])
	_0_b = np.zeros([1,4])
	_1_W = np.zeros([4,2])
	_1_b = np.zeros([1,2])
	

	#Step value and batch size
	learning_rate = 0.1
	mini_batch_size = np.shape(features)[0]
	for i in range(step):
		#Forward propagation
		s0 = features *weightsLayer1
		s0 +=  _0_b
		
		#Apply sigmoid to layer 1
		sigmoided = sigmoid(s0)
		#Feed into the next layer

		s1 = sigmoided * weightsLayer2
		s1 += _1_b
		
		#Apply softmax function; second activation. Turn numbers into probabilities
		softmaxed = softmax(s1)

		delta = softmaxed - labels

		
		#Delta weighted from weightsLayer2, then transpose matrix
		epsilonNext = (weightsLayer2 * delta.T).T

		#Sigmoid derivative
		dLdz = np.multiply(sigmoided, (1-sigmoided))
		
		#Backpropagation
		backpropagation = np.multiply(dLdz, epsilonNext)
		
		#Update gradients and weights
		_0_W = features.T * backpropagation + _0_W
		_1_W = (sigmoided.T * delta) + _1_W
		_0_b = _0_b - (learning_rate * np.sum(backpropagation, 0))
		_1_b = _1_b - (learning_rate * np.sum(delta, 0))

		weightsLayer1 = weightsLayer1 - (learning_rate * _0_W)
		weightsLayer2 = weightsLayer2 - (learning_rate * _1_W)

	return softmaxed
	
print(truthTable_neural(1500))