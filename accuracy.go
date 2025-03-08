package main

import "gonum.org/v1/gonum/mat"

func argMaxColumn(M *mat.Dense, col int) int {
	rows, _ := M.Dims()
	maxIndex := 0
	maxVal := M.At(0, col)
	for i := 1; i < rows; i++ {
		v := M.At(i, col)
		if v > maxVal {
			maxVal = v
			maxIndex = i
		}
	}
	return maxIndex
}

func computeAccuracy(A, Y *mat.Dense) float64 {
	_, cols := A.Dims()
	correct := 0
	for i := 0;i < cols; i++ {
		predIndex := argMaxColumn(A, i)
		trueIndex := argMaxColumn(Y, i)
		if predIndex == trueIndex {
			correct++
		}
	}
	return float64(correct) / float64(cols)
}