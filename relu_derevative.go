package main

func reluDerevative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}