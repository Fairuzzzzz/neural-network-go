package main

import (
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

func neuralNetwork(input, output, hidden int) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	// First Layer
	W1 := mat.NewDense(hidden, input, nil)
	b1 := mat.NewDense(hidden, 1, nil)

	// Second Layer
	W2 := mat.NewDense(output, hidden, nil)
	b2 := mat.NewDense(output, 1, nil)

	// Initialize weights with small random values
	for i := 0; i < hidden; i++ {
		for j := 0; j < input; j++ {
			W1.Set(i, j, rand.Float64()*0.01)
		}
		b1.Set(i, 0, 0.0)
	}

	for i := 0; i < output; i++ {
		for j := 0; j < hidden; j++ {
			W2.Set(i, j, rand.Float64()*0.01)
		}
		b2.Set(i, 0, 0.0)
	}
	return W1, b1, W2, b2
}
