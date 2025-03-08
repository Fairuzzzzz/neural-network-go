package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func loadCSV(filename string) (*mat.Dense, *mat.Dense) {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Jumlah Sample
	numSample := len(lines) - 1
	numFeatures := len(lines[0]) - 1

	// Matrix untuk input (x) dan label (y)
	X := mat.NewDense(numFeatures, numSample, nil)
	Y := mat.NewDense(10, numSample, nil) // One-hot encoding (10 label)

	for i, line := range lines[1:] {
		label, _ := strconv.Atoi(line[0])
		Y.Set(label, i, 1.0)

		for j := 1; j <= numFeatures; j++ {
			val, _ := strconv.Atoi(line[j])
			X.Set(j-1, i, float64(val)/255.0)
		}
	}

	return X, Y
}
