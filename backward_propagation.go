package main

import "gonum.org/v1/gonum/mat"

func backwardPropagation(X, Y, Z, W, b, A *mat.Dense, learningRate float64) {
	// Batch Size
	m := float64(X.RawMatrix().Cols)

	// Hitung dA
	dA := mat.NewDense(A.RawMatrix().Rows, A.RawMatrix().Cols, nil)
	dA.Sub(A, Y) // dA = A - Y

	// Hitung dZ = dA * ReLU(derevative)(Z)
	dZ := mat.NewDense(Z.RawMatrix().Rows, Z.RawMatrix().Cols, nil)
	dZ.Apply(func(i, j int, v float64) float64 {
		return dA.At(i, j) * reluDerevative(v)
	}, Z)

	// Hitung gradients
	Xt := mat.DenseCopyOf(X.T())
	dW := mat.NewDense(W.RawMatrix().Rows, W.RawMatrix().Cols, nil)
	dW.Mul(dZ, Xt)
	dW.Scale(1/m, dW)

	db := mat.NewDense(b.RawMatrix().Rows, 1, nil)
	for i := 0; i < dZ.RawMatrix().Rows; i++ {
		sum := 0.0
		for j := 0; j < dZ.RawMatrix().Cols; j++ {
			sum += dZ.At(i, j)
		}
		db.Set(i, 0, sum/m)
	}

	// Update Weight dan Bias
	dW.Scale(learningRate, dW)
	db.Scale(learningRate, db)
	W.Sub(W, dW)
	b.Sub(b, db)
}
