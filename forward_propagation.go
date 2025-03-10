package main

import "gonum.org/v1/gonum/mat"

func forwardPropagation(X, W1, b1, W2, b2 *mat.Dense) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	// First Layer
	Z1 := mat.NewDense(W1.RawMatrix().Rows, X.RawMatrix().Cols, nil)
	Z1.Mul(W1, X)

	// Add Bias
	rows1, cols1 := Z1.Dims()
	for i := 0; i < rows1; i++ {
		biasValue := b1.At(i, 0)
		for j := 0; j < cols1; j++ {
			Z1.Set(i, j, Z1.At(i, j)+biasValue)
		}
	}

	// Apply ReLU
	A1 := mat.NewDense(Z1.RawMatrix().Rows, Z1.RawMatrix().Cols, nil)
	A1.Apply(func(i, j int, v float64) float64 {
		return relu(v)
	}, Z1)

	// Second Layer (softmax)
	Z2 := mat.NewDense(W2.RawMatrix().Rows, A1.RawMatrix().Cols, nil)
	Z2.Mul(W2, A1)

	// Add Bias
	rows2, cols2 := Z2.Dims()
	for i := 0; i < rows2; i++ {
		biasValue := b2.At(i, 0)
		for j := 0; j < cols2; j++ {
			Z2.Set(i, j, Z2.At(i, j)+biasValue)
		}
	}

	// Apply Softmax
	A2 := mat.NewDense(Z2.RawMatrix().Rows, Z2.RawMatrix().Cols, nil)
	for j := 0; j < cols2; j++ {
		colData := make([]float64, rows2)
		for i := 0; i < rows2; i++ {
			colData[i] = Z2.At(i, j)
		}

		softmaxOuput := softmax(colData)

		// Put result back into A2
		for i := 0; i < rows2; i++ {
			A2.Set(i, j, softmaxOuput[i])
		}
	}

	return A1, Z1, A2, Z2
}
