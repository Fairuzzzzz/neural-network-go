package main

import "gonum.org/v1/gonum/mat"

func computeLoss(A, Y *mat.Dense) float64 {
	_, cols := A.Dims()
	sum := 0.0
	rows, _ := A.Dims()
	for i := 0; i < rows; i++ {
		for j := 0;j < cols;j++ {
			diff := A.At(i, j) - Y.At(i, j)
			sum += diff * diff
		}
	}
	return sum / float64(cols)
}