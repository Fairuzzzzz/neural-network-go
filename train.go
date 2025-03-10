package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func train(X, Y *mat.Dense, learningRate float64, epochs int) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	inputSize := X.RawMatrix().Rows
	hiddenSize := 128
	outputSize := Y.RawMatrix().Rows

	W1, b1, W2, b2 := neuralNetwork(inputSize, outputSize, hiddenSize)

	// Training Loop
	for i := 0; i < epochs; i++ {
		// Forward Propagation
		A1, Z1, A2, Z2 := forwardPropagation(X, W1, b1, W2, b2)

		// Backward Propagation
		backwardPropagation(X, Y, Z1, W1, b1, A1, Z2, W2, b2, A2, learningRate)

		if i % 100 == 0 {
			loss := computeLoss(A2, Y)
			accuracy := computeAccuracy(A2, Y)
			fmt.Printf("Epoch %d: Loss = %f, Accuracy = %f\n", i, loss, accuracy)
		}
	}

	return W1, b1, W2, b2
}