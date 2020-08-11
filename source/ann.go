package main

import "fmt"

func main() {
	fmt.Println("Starting ANN...")
	n := network{784, 200, 10, 0.3}
	fmt.Println("Input nodes for verification: ", n.inputNodes)
	initialize()
}

type network struct {
	inputNodes, hiddenNodes, outputNodes uint16
	learningRate                         float32
}

func (n *network) initialize() {
	//random wih and who needs to be added & other init stuff (check the other code)
}
