package main

import "gonum.org/v1/gonum/mat"

func predict(X, W1, b1, W2, b2 *mat.Dense) *mat.Dense {
	_, _, A2, _ := forwardPropagation(X, W1, b1, W2, b2)
	return A2
}
