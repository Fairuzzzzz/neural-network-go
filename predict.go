package main

import "gonum.org/v1/gonum/mat"

func predict(X, W, b *mat.Dense) *mat.Dense {
	A, _ := forwardPropagation(X, W, b)
	return A
}
