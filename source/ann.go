package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	fmt.Println("Starting ANN...")
	n := Network{inputNodes: 784, hiddenNodes: 200, outputNodes: 10, learningRate: 0.3}
	fmt.Println("Input nodes for verification: ", n.inputNodes)
	n.initialize()
}

//Network is main ann attributes
type Network struct {
	inputNodes, hiddenNodes, outputNodes uint16
	learningRate                         float32
	weightedHiddenInputs                 [][]float64
	weightedHiddenOutputs                [][]float64
}

func (n *Network) initialize() {
	//random wih and who needs to be added & other init stuff (check the other code)
	rand.Seed(time.Now().UnixNano())

	n.weightedHiddenInputs = make([][]float64, n.inputNodes)
	for index := range n.weightedHiddenInputs {
		n.weightedHiddenInputs[index] = make([]float64, n.hiddenNodes)
		for index2 := range n.weightedHiddenInputs[index] {
			n.weightedHiddenInputs[index][index2] = (-0.99 + rand.Float64()*(0.99 - -0.99))
		}
	}

	n.weightedHiddenOutputs = make([][]float64, n.hiddenNodes)
	for index := range n.weightedHiddenOutputs {
		n.weightedHiddenOutputs[index] = make([]float64, n.outputNodes)
		for index2 := range n.weightedHiddenOutputs[index] {
			n.weightedHiddenOutputs[index][index2] = (-0.99 + rand.Float64()*(0.99 - -0.99))
		}
	}
	fmt.Println("Completed initial slice creation")

	//***testing purposes
	for index := range n.weightedHiddenInputs {
		for _, value := range n.weightedHiddenInputs[index] {
			fmt.Println("Value at index '", index, "' --|| ", value)
		}
	}
	//end testing code***

}
