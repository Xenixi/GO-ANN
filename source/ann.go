package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/zieckey/goini"
)

func main() {
	fmt.Println("Starting ANN...")

	if _, e := os.Stat("settings.ini"); os.IsNotExist(e) {
		os.Create("settings.ini")
		fmt.Println("settings.ini not found. \n Created.")
	} else {
		fmt.Println(" - settings.ini exists!")
	}

	iniFile := goini.New()
	if err := iniFile.ParseFile("settings.ini"); err != nil {
		fmt.Println("Failed to parse ini. Aborting...")
		os.Exit(0)
	}
	getInputNodes, _ := iniFile.Get("input-nodes")
	inputNodes64, _ := strconv.ParseUint(getInputNodes, 0, 16)

	getHiddenNodes, _ := iniFile.Get("hidden-nodes")
	hiddenNodes64, _ := strconv.ParseUint(getHiddenNodes, 0, 16)

	getOutputNodes, _ := iniFile.Get("output-nodes")
	outputNodes64, _ := strconv.ParseUint(getOutputNodes, 0, 16)

	getLearningRate, _ := iniFile.Get("learning-rate")
	learningRate64, _ := strconv.ParseFloat(getLearningRate, 16)

	n := Network{inputNodes: uint16(inputNodes64), hiddenNodes: uint16(hiddenNodes64), outputNodes: uint16(outputNodes64), learningRate: float32(math.Round(learningRate64*10) / 10)}
	fmt.Println("---------------------------------------\nInput nodes for verification:", n.inputNodes)
	fmt.Println("Hidden nodes for verification:", n.hiddenNodes)
	fmt.Println("Output nodes for verification:", n.outputNodes)
	fmt.Println("Learning rate for verification:", n.learningRate, "\n---------------------------------------")

	n.initialize()
}

//Network is main ann attributes
type Network struct {
	inputNodes, hiddenNodes, outputNodes uint16
	learningRate                         float32
	weightedHiddenInputs                 [][]float64
	weightedHiddenOutputs                [][]float64
}

func activation(x float64) float64 {
	return (1 / (1 + math.Pow(math.E, -x)))
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
	fmt.Println(" - Completed initial slice creation")
	//***testing purposes
	/*	for index := range n.weightedHiddenInputs {
			for _, value := range n.weightedHiddenInputs[index] {
				fmt.Println("Value at index '", index, "' --|| ", value)
			}
		}
	*/
	//end testing code***

	///////////////////////////////

}
