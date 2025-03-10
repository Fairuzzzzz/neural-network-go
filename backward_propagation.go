package main

import "gonum.org/v1/gonum/mat"

func backwardPropagation(X, Y, Z1, W1, b1, A1, Z2, W2, b2, A2 *mat.Dense, learningRate float64) {
	// Batch Size
	m := float64(X.RawMatrix().Cols)

	// Output layer gradients
	dZ2 := mat.NewDense(A2.RawMatrix().Rows, A2.RawMatrix().Cols, nil)
	dZ2.Sub(A2, Y)

	// Update W2 gradients
	dW2 := mat.NewDense(W2.RawMatrix().Rows, W2.RawMatrix().Cols, nil)
	dW2.Mul(dZ2, A1.T())
	dW2.Scale(1/m, dW2)

	// Update b2 gradients
	db2 := mat.NewDense(b2.RawMatrix().Rows, b2.RawMatrix().Cols, nil)
	for i := 0; i < dZ2.RawMatrix().Rows; i++ {
		sum := 0.0
		for j := 0; j < dZ2.RawMatrix().Cols;j++ {
			sum += dZ2.At(i, j)
		}
		db2.Set(i, 0, sum/m)
	}

	// Hidden layer gradients
	dA1 := mat.NewDense(A1.RawMatrix().Rows, A1.RawMatrix().Cols, nil)
	dA1.Mul(W2.T(), dZ2)

	// Apply ReLU derivative
	dZ1 := mat.NewDense(Z1.RawMatrix().Rows, Z1.RawMatrix().Cols, nil)
	dZ1.Apply(func(i, j int, v float64) float64 {
		return dA1.At(i, j) * reluDerevative(v)
	}, Z1)

	// Update W1 gradients
	dW1 := mat.NewDense(W1.RawMatrix().Rows, W1.RawMatrix().Cols, nil)
	dW1.Mul(dZ1, X.T())
	dW1.Scale(1/m, dW1)

	// Update b1 gradients
	db1 := mat.NewDense(b1.RawMatrix().Rows, b1.RawMatrix().Cols, nil)
	for i := 0;i < dZ1.RawMatrix().Rows;i++ {
		sum := 0.0
		for j := 0; j < dZ1.RawMatrix().Cols; j++ {
			sum += dZ1.At(i, j)
		}
		db1.Set(i, 0, sum/m)
	}

	// Update parameters
	dW1.Scale(learningRate, dW1)
	dW2.Scale(learningRate, dW2)
	db1.Scale(learningRate, db1)
	db2.Scale(learningRate, db2)
	
	W1.Sub(W1, dW1)
	W2.Sub(W2, dW2)
	b1.Sub(b1, db1)
	b2.Sub(b2, db2)

	
}
