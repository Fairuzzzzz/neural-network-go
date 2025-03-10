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
	W1, b1, W2, b2 := train(X, Y, learningRate, epoch)

	// Predict
	A := predict(X, W1, b1, W2, b2)
	fmt.Println("Sample Prediction: ", mat.Formatted(A))

	// Calculate final accuracy
	acc := computeAccuracy(A, Y)
	fmt.Printf("Final Accuracy: %f\n", acc)
}
