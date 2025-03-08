package main

import "gonum.org/v1/gonum/mat"

func forwardPropagation(X, W, b *mat.Dense) (*mat.Dense, *mat.Dense) {
	Z := mat.NewDense(X.RawMatrix().Rows, X.RawMatrix().Cols, nil)

	Z.Mul(W, X)
	Z.Add(Z, b)

	// ReLU Activation
	A := mat.NewDense(X.RawMatrix().Rows, X.RawMatrix().Cols, nil)
	A.Apply(func(i, j int, v float64) float64 {
		return relu(v)
	}, Z)
	return A, Z
}