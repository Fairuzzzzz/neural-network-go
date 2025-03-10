package main

import "math"

func softmax(x []float64) []float64 {
	result := make([]float64, len(x))
	var sum float64 = 0.0

	var max float64 = x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	for i, v := range x {
		result[i] = math.Exp(v - max)
		sum += result[i]
	}

	for i := range result {
		result[i] /= sum
	}
	return result
}