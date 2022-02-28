package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
)

type TrainingSetLabelFiles struct {
	MagicNumber   int32
	NumberOfItems int32
	Labels        []byte // The labels values are 0 to 9.
}

type TrainingSetImageFiles struct {
	MagicNumber     int32
	NumberOfImages  int32
	NumberOfRows    int32
	NumberOfColumns int32
	Pixels          []byte // Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
}

type LayerMap map[int](map[int]float64)

type ComputedResult struct {
	NodesL0 [numOfPixels]float64 // pixels
	NodesL1 [16]float64
	NodesL2 [16]float64
	NodesL3 [10]float64 // results
}

type Network struct {
	LMaps  [3]LayerMap
	Biases [4]([]float64)
}

const (
	numOfPixels    = 28 * 28
	learning_const = 0.1
)

// don't forget the bias

func main() {
	fileTestData, err := os.Open("t10k-images.idx3-ubyte")
	check(err)
	defer fileTestData.Close()

	fileTestDataLabels, err := os.Open("t10k-labels.idx1-ubyte")
	check(err)
	defer fileTestDataLabels.Close()

	testFile := parseImageFile(fileTestData)
	labelFile := parseLabelFile(fileTestDataLabels)

	network := initNetwork()

	for imageNum := 0; imageNum < 10; imageNum++ { // image 1 shows a '2'
		result := calculateResult(getImage(testFile, imageNum), network)
		fmt.Println(cost(result.NodesL3, int(labelFile.Labels[imageNum])))
		deltas := calcAllDeltas(network, result, int(labelFile.Labels[imageNum]))
		learn(network, result, deltas)
	}

	// for i := 0; i < 12; i++ {
	// 	showPicture(testFile, i*11)
	// 	ratings := calculateResult(getImage(testFile, i), network)
	// 	showResult(ratings)
	// 	fmt.Printf("cost: %v\n", cost(ratings, int(labelFile.Labels[i])))
	// }
}

func learn(network Network, result ComputedResult, deltas map[int](map[int]float64)) {
	// lmap2
	for i := 0; i < 16; i++ {
		for j := 0; j < 10; j++ {
			network.LMaps[2][i][j] += (-1) * learning_const * result.NodesL2[i] * deltas[3][j]
		}
	}

	// lmap1
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			network.LMaps[1][i][j] += (-1) * learning_const * result.NodesL1[i] * deltas[2][j]
		}
	}

	// lmap0
	for i := 0; i < numOfPixels; i++ {
		for j := 0; j < 16; j++ {
			network.LMaps[0][i][j] += (-1) * learning_const * result.NodesL0[i] * deltas[1][j]
		}
	}

}

func calcAllDeltas(network Network, result ComputedResult, label int) map[int](map[int]float64) {
	deltas := make(map[int](map[int]float64)) // deltas[layer][nrOfNode]

	indik := func(label int, j int) int {
		if label == j {
			return 1
		} else {
			return 0
		}
	}

	for layer := 3; layer >= 0; layer += -1 {
		switch layer {
		case 3: // outputLayer
			for neuron := 0; neuron < 10; neuron++ {
				sigma := ((result.NodesL3[neuron] - float64(indik(label, neuron))) * result.NodesL3[neuron] * (1 - result.NodesL3[neuron]))
				deltas[3][neuron] = sigma
			}

		case 2:
			for neuronL := 0; neuronL < 16; neuronL++ {
				var someSum float64
				for neuronR := 0; neuronR < 10; neuronR++ {
					someSum += network.LMaps[2][neuronL][neuronR] * deltas[3][neuronR]
				}

				sigma := someSum * result.NodesL2[neuronL] * (1 - result.NodesL2[neuronL])
				deltas[2][neuronL] = sigma
			}

		case 1:
			for neuronL := 0; neuronL < 16; neuronL++ {
				var someSum float64
				for neuronR := 0; neuronR < 16; neuronR++ {
					someSum += network.LMaps[1][neuronL][neuronR] * deltas[2][neuronR]
				}

				sigma := someSum * result.NodesL1[neuronL] * (1 - result.NodesL1[neuronL])
				deltas[1][neuronL] = sigma
			}

		case 0:
			for neuronL := 0; neuronL < numOfPixels; neuronL++ {
				var someSum float64
				for neuronR := 0; neuronR < 16; neuronR++ {
					someSum += network.LMaps[0][neuronL][neuronR] * deltas[1][neuronR]
				}

				sigma := someSum * result.NodesL0[neuronL] * (1 - result.NodesL0[neuronL])
				deltas[0][neuronL] = sigma
			}

		default:
			panic("Invalid layer nr.")
		}

	}

	return deltas
}

// func derivate(network Network, neuronL int, neuronR int, layer int, result ComputedResult, label int) float64 {
// 	indik := func(label int, j int) int {
// 		if label == j {
// 			return 1
// 		} else {
// 			return 0
// 		}
// 	}

// 	switch layer {
// 	case 3: // outputLayer
// 		sigma := ((result.NodesL3[neuronR] - float64(indik(label, neuronR))) * result.NodesL3[neuronR] * (1 - result.NodesL3[neuronR]))
// 		return result.NodesL2[neuronL] * sigma

// 	case 2:
// 		var someSum float64
// 		for _, weight_l := range network.LMaps[2][neuronR] {
// 			someSum += weight_l * sigmaForOutputNeurons(network, neuronL, neuronR, label, result)
// 		}

// 		sigma := someSum * result.NodesL3[neuronR] * (1 - result.NodesL3[neuronR])
// 		return result.NodesL2[neuronL] * sigma

// 	case 1:
// 		var someSum float64
// 		for _, weight_l := range network.LMaps[1][neuronR] {
// 			someSum += weight_l * sigmaForOutputNeurons(network, neuronL, neuronR, label, result)
// 		}

// 		sigma := someSum * result.NodesL2[neuronR] * (1 - result.NodesL2[neuronR])
// 		return result.NodesL1[neuronL] * sigma

// 	default:
// 		panic("Invalid layer nr.")
// 	}

// }

// func derivativeForInnerNeurons(network Network, i int, j int, label int, result ComputedResult) float64 {
// 	return result.NodesL2[i] * sigmaForInnerNeurons(network, i, j, label, result)
// }

// func derivativeForOutputNeurons(network Network, i int, j int, label int, result ComputedResult) float64 {
// 	return result.NodesL2[i] * sigmaForOutputNeurons(network, i, j, label, result)
// }

// func sigmaForInnerNeurons(network Network, i int, j int, label int, result ComputedResult) float64 {
// 	// first only for layer2 out of simplicity
// 	var someSum float64
// 	for _, weight_l := range network.LMaps[2][j] {
// 		someSum += weight_l * sigmaForOutputNeurons(network, i, j, label, result)
// 	}

// 	return someSum * result.NodesL3[j] * (1 - result.NodesL3[j])
// }

// func sigmaForOutputNeurons(network Network, i int, j int, label int, result ComputedResult) float64 {
// 	var indikLabel int
// 	if label == j {
// 		indikLabel = 1
// 	} else {
// 		indikLabel = 0
// 	}

// 	return ((result.NodesL3[j] - float64(indikLabel)) * result.NodesL3[j] * (1 - result.NodesL3[j]))
// }

func calculateResult(image []byte, network Network) ComputedResult {
	var results ComputedResult

	// layer0 -> layer1
	for i := 0; i < 16; i++ {
		for j := 0; j < numOfPixels; j++ {
			results.NodesL1[i] += (network.LMaps[0])[j][i] * (float64(int(image[j])) / 255)
		}
		results.NodesL1[i] = sigmoid(results.NodesL1[i] - network.Biases[0][i])
	}

	// layer1 -> layer2
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			results.NodesL2[i] += (network.LMaps[1])[j][i] * results.NodesL1[j]
		}
		results.NodesL2[i] = sigmoid(results.NodesL2[i] - network.Biases[1][i])
	}

	// layer2 -> layer3
	for i := 0; i < 10; i++ {
		for j := 0; j < 16; j++ {
			results.NodesL3[i] += (network.LMaps[2])[j][i] * results.NodesL2[j]
		}
		results.NodesL3[i] = sigmoid(results.NodesL3[i] - network.Biases[2][i])
	}

	return results
}

func cost(result [10]float64, label int) float64 {
	var costVal float64
	for i := 0; i < 10; i++ {
		if i == label {
			costVal += math.Pow((result[i] - 1), 2)
		} else {
			costVal += math.Pow((result[i] - 0), 2)
		}
	}

	return costVal
}

func initNetwork() Network {
	// initialize the maps between layers randomly
	// initialize all biases to zero
	var network Network

	network.LMaps[0] = make(LayerMap)
	for i := 0; i < numOfPixels; i++ {
		network.LMaps[0][i] = make(map[int]float64, 0)
	}

	network.LMaps[1] = make(LayerMap)
	for i := 0; i < 16; i++ {
		network.LMaps[1][i] = make(map[int]float64, 0)
	}

	network.LMaps[2] = make(LayerMap)
	for i := 0; i < 16; i++ {
		network.LMaps[2][i] = make(map[int]float64, 0)
	}

	network.Biases[0] = make([]float64, 16)
	network.Biases[1] = make([]float64, 16)
	network.Biases[2] = make([]float64, 10)

	// lMap0
	for i := 0; i < numOfPixels; i++ {
		for j := 0; j < 16; j++ {
			(network.LMaps[0])[i][j] = rand.Float64()
		}
	}

	// lMap1
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			(network.LMaps[1])[i][j] = rand.Float64()
		}
	}

	// lMap2
	for i := 0; i < 16; i++ {
		for j := 0; j < 10; j++ {
			(network.LMaps[2])[i][j] = rand.Float64()
		}
	}

	// bias0
	for i := 0; i < 16; i++ {
		// network.Biases[0][i] = float64(10)
		network.Biases[0][i] = rand.Float64()
	}

	// bias1
	for i := 0; i < 16; i++ {
		// network.Biases[1][i] = float64(10)
		network.Biases[1][i] = rand.Float64()
	}

	// bias2
	for i := 0; i < 10; i++ {
		// network.Biases[2][i] = float64(0)
		network.Biases[2][i] = rand.Float64()
	}

	return network
}

func getImage(archive TrainingSetImageFiles, fileNum int) []byte { // return the pixels of the specific image
	return archive.Pixels[fileNum*numOfPixels : (fileNum+1)*numOfPixels]
}

func showPicture(archive TrainingSetImageFiles, fileNum int) {
	if fileNum > int(archive.NumberOfImages) {
		panic("fileNum is to high")
	}

	image := getImage(archive, fileNum)

	printByte := func(b byte) {
		i := int((int(b) * 23 / 255)) + 232

		fmt.Printf("\033[48;5;%dm ", i)
		fmt.Printf(" ")
		fmt.Printf("\033[0m")
	}

	i := 0
	for i < numOfPixels {
		if i%int(archive.NumberOfColumns) == 0 {
			fmt.Printf("\n")
		}
		printByte(image[i])

		i++
	}
	fmt.Printf("\n")
}

func showResult(result [10]float64) {
	for i := 0; i < 10; i++ {
		fmt.Printf("Value for %v:\t %9.3f\n", i, result[i])
	}
}

func parseImageFile(file *os.File) TrainingSetImageFiles {
	var testFile TrainingSetImageFiles
	err := binary.Read(file, binary.BigEndian, &testFile.MagicNumber)
	check(err)
	err = binary.Read(file, binary.BigEndian, &testFile.NumberOfImages)
	check(err)
	err = binary.Read(file, binary.BigEndian, &testFile.NumberOfRows)
	check(err)
	err = binary.Read(file, binary.BigEndian, &testFile.NumberOfColumns)
	check(err)

	pixels := make([]byte, int(testFile.NumberOfImages)*numOfPixels)

	_, err = file.Read(pixels)
	check(err)

	testFile.Pixels = pixels

	return testFile
}

func parseLabelFile(file *os.File) TrainingSetLabelFiles {
	var labelFile TrainingSetLabelFiles
	err := binary.Read(file, binary.BigEndian, &labelFile.MagicNumber)
	check(err)
	err = binary.Read(file, binary.BigEndian, &labelFile.NumberOfItems)
	check(err)

	labels := make([]byte, labelFile.NumberOfItems)
	_, err = file.Read(labels)
	check(err)

	labelFile.Labels = labels

	return labelFile
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(x*(-1)))
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
