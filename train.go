package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func train(X, Y *mat.Dense, learningRate float64, epochs int) (*mat.Dense, *mat.Dense) {
	inputSize := X.RawMatrix().Rows
	outputSize := Y.RawMatrix().Rows

	W, b := neuralNetwork(inputSize, outputSize)
	for i := 0; i < epochs; i++ {
		A, Z := forwardPropagation(X, W, b)
		backwardPropagation(X, Y, Z, W, b, A, learningRate)

		// Print Lose 
		if i%100 == 0 {
			loss := computeLoss(A, Y)
			accuracy := computeAccuracy(A, Y)
			fmt.Printf("Epoch %d: Loss = %f, Accuracy = %f\n", i, loss, accuracy)
		}
	}
	return W, b
}