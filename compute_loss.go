package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func computeLoss(A, Y *mat.Dense) float64 {
	rows, cols := A.Dims()
	sum := 0.0
	epsilion := 1e-15

	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			if Y.At(i, j) > 0 {
				pred := math.Max(A.At(i, j), epsilion)
				sum -= Y.At(i, j) * math.Log(pred)
			}
		}
	}

	return sum / float64(cols)
}
