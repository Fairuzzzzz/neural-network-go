package main

import "gonum.org/v1/gonum/mat"

func forwardPropagation(X, W, b *mat.Dense) (*mat.Dense, *mat.Dense) {
	Z := mat.NewDense(W.RawMatrix().Rows, X.RawMatrix().Cols, nil)

	Z.Mul(W, X)

	rows, cols := Z.Dims()
	for i := 0; i < rows; i++ {
		biasValue := b.At(i, 0)
		for j := 0; j < cols; j++ {
			Z.Set(i, j, Z.At(i, j)+biasValue)
		}
	}

	// ReLU Activation
	A := mat.NewDense(Z.RawMatrix().Rows, Z.RawMatrix().Cols, nil)
	A.Apply(func(i, j int, v float64) float64 {
		return relu(v)
	}, Z)
	return A, Z
}
