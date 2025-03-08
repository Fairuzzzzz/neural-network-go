package main

import (
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

func neuralNetwork(input, output int) (*mat.Dense, *mat.Dense) {
	W := mat.NewDense(output, input, nil)
	b := mat.NewDense(output, 1, nil)

	for i := 0; i < output; i++ {
		for j := 0; j < input; j++ {
			W.Set(i, j, rand.Float64()*0.01)
		}
		b.Set(i, 0, 0.0)
	}
	return W, b
}
