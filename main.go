package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	X, Y := loadCSV("data/train.csv")

	epoch := 1000
	learningRate := 0.01

	// Train
	W, b := train(X, Y, learningRate, epoch)

	// Predict
	A := predict(X, W, b)
	fmt.Println("Sample Prediction: ", mat.Formatted(A))
}
