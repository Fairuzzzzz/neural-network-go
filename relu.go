package main

import "math"

func relu(x float64) float64 {
	return math.Max(0, x)
}
