package main

import (
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

type TrainingLabelFiles struct {
	MagicNumber   int32
	NumberOfItems int32
	Labels        []byte // The labels values are 0 to 9.
}

type TrainingImageFiles struct {
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

func main() {
	fileTrainingData, err := os.Open("train-images.idx3-ubyte") // contains 60000 pictures
	check(err)
	defer fileTrainingData.Close()

	fileTrainingDataLabels, err := os.Open("train-labels.idx1-ubyte")
	check(err)
	defer fileTrainingDataLabels.Close()

	trainingFile := parseImageFile(fileTrainingData)
	// TraininglabelFile := parseLabelFile(fileTrainingDataLabels)

	network := loadNetwork("network1.gob")

	rand.Seed(time.Now().Unix())
	fileNum := rand.Intn(59999)
	result := calculateResult(getImage(trainingFile, fileNum), network)

	showPicture(trainingFile, fileNum)
	showResult(result.NodesL3)

	// network := initNetwork()

	// trainNetwork(&network, trainingFile, TraininglabelFile)

	// saveNetwork(&network)
}

func trainNetwork(network *Network, trainingFile TrainingImageFiles, trainingLabels TrainingLabelFiles) {
	avgError := float64(0)
	for imageNum := 0; imageNum < 60000; imageNum++ { // image 1 shows a '2'
		label := int(trainingLabels.Labels[imageNum])
		result := calculateResult(getImage(trainingFile, imageNum), network)

		avgError = addToAvg(imageNum%11, avgError, cost(result.NodesL3, label))

		deltas := calcAllDeltas(network, &result, label)
		learn(network, result, deltas)

		if imageNum%2000-5 == 0 {
			fmt.Println("avgError:", avgError)
			avgError = 0

			showPicture(trainingFile, imageNum)
			fmt.Println("imageNum:", imageNum)
			showResult(result.NodesL3)
			fmt.Println("Label:", label)
		}
	}
}

func loadNetwork(filename string) *Network {
	file, err := os.Open(filename)
	check(err)
	defer file.Close()

	decoder := gob.NewDecoder(file)

	var network Network
	err = decoder.Decode(&network)
	check(err)

	return &network
}

func saveNetwork(network *Network) {
	file, err := os.Create("network1.gob")
	check(err)
	defer file.Close()

	encoder := gob.NewEncoder(file)

	err = encoder.Encode(network)
	check(err)
}

func addToAvg(sampleSize int, currentAvg float64, newVal float64) float64 {
	ret := (float64(sampleSize)*currentAvg + newVal) / float64(sampleSize+1)
	return ret
}

func learn(network *Network, result ComputedResult, deltas map[int](map[int]float64)) {
	var currChange float64
	// lmap2
	for i := 0; i < 16; i++ {
		for j := 0; j < 10; j++ {
			currChange = result.NodesL2[i] * deltas[3][j]
			network.LMaps[2][i][j] += (-1) * learning_const * currChange
		}
	}

	// lmap1
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			currChange = result.NodesL1[i] * deltas[2][j]
			network.LMaps[1][i][j] += (-1) * learning_const * currChange
		}
	}

	// lmap0
	for i := 0; i < numOfPixels; i++ {
		for j := 0; j < 16; j++ {
			currChange = result.NodesL0[i] * deltas[1][j]
			network.LMaps[0][i][j] += (-1) * learning_const * currChange
		}
	}

	for layer := 1; layer < 4; layer++ {
		switch layer {
		case 1:
			for i := 0; i < 16; i++ {
				network.Biases[0][i] += (-1) * learning_const * deltas[1][i]
			}
		case 2:
			for i := 0; i < 16; i++ {
				network.Biases[1][i] += (-1) * learning_const * deltas[2][i]
			}
		case 3:
			for i := 0; i < 10; i++ {
				network.Biases[2][i] += (-1) * learning_const * deltas[3][i]
			}
		}
	}
}

func calcAllDeltas(network *Network, result *ComputedResult, label int) map[int](map[int]float64) {
	deltas := make(map[int](map[int]float64)) // deltas[layer][nrOfNode]
	for j := 1; j < 4; j++ {
		deltas[j] = make(map[int]float64)
	}

	indik := func(label int, j int) int {
		if label == j {
			return 1
		} else {
			return 0
		}
	}

	for layer := 3; layer > 0; layer += -1 {
		switch layer {
		case 3: // outputLayer
			for neuron := 0; neuron < 10; neuron++ {
				sigma := ((result.NodesL3[neuron] - float64(indik(label, neuron))) * result.NodesL3[neuron] * (1 - result.NodesL3[neuron]))
				deltas[3][neuron] = sigma
			}

		case 2:
			for neuronL := 0; neuronL < 16; neuronL++ {
				var someSum float64
				someSum = 0
				for neuronR := 0; neuronR < 10; neuronR++ {
					someSum += network.LMaps[2][neuronL][neuronR] * deltas[3][neuronR]
				}

				sigma := someSum * result.NodesL2[neuronL] * (1 - result.NodesL2[neuronL])
				deltas[2][neuronL] = sigma
			}

		case 1:
			for neuronL := 0; neuronL < 16; neuronL++ {
				var someSum float64
				someSum = 0
				for neuronR := 0; neuronR < 16; neuronR++ {
					someSum += network.LMaps[1][neuronL][neuronR] * deltas[2][neuronR]
				}

				sigma := someSum * result.NodesL1[neuronL] * (1 - result.NodesL1[neuronL])
				deltas[1][neuronL] = sigma
			}

		// case 0:
		// 	for neuronL := 0; neuronL < numOfPixels; neuronL++ {
		// 		var someSum float64
		// 		for neuronR := 0; neuronR < 16; neuronR++ {
		// 			someSum += network.LMaps[0][neuronL][neuronR] * deltas[1][neuronR]
		// 		}

		// 		sigma := someSum * result.NodesL0[neuronL] * (1 - result.NodesL0[neuronL])
		// 		deltas[0][neuronL] = sigma
		// 	}

		default:
			panic("Invalid layer nr.")
		}

	}

	return deltas
}

func calculateResult(image []byte, network *Network) ComputedResult {
	var results ComputedResult

	for i := 0; i < numOfPixels; i++ {
		results.NodesL0[i] = float64(image[i]) / 255
	}

	// layer0 -> layer1
	for i := 0; i < 16; i++ {
		var sum float64
		for j := 0; j < numOfPixels; j++ {
			sum += (network.LMaps[0])[j][i] * results.NodesL0[j]
		}
		results.NodesL1[i] = sigmoid(sum - network.Biases[0][i])
	}

	// layer1 -> layer2
	for i := 0; i < 16; i++ {
		var sum float64
		for j := 0; j < 16; j++ {
			sum += (network.LMaps[1])[j][i] * results.NodesL1[j]
		}
		results.NodesL2[i] = sigmoid(sum - network.Biases[1][i])
	}

	// layer2 -> layer3
	for i := 0; i < 10; i++ {
		var sum float64
		for j := 0; j < 16; j++ {
			sum += (network.LMaps[2])[j][i] * results.NodesL2[j]
		}
		results.NodesL3[i] = sigmoid(sum - network.Biases[2][i])
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

	// return sigmoid(costVal)
	return costVal / 2
}

func initNetwork() Network {
	// initialize the maps between layers randomly
	// initialize all biases to zero
	var network Network

	network.LMaps[0] = make(LayerMap)
	for i := 0; i < numOfPixels; i++ {
		network.LMaps[0][i] = make(map[int]float64)
	}

	network.LMaps[1] = make(LayerMap)
	for i := 0; i < 16; i++ {
		network.LMaps[1][i] = make(map[int]float64)
	}

	network.LMaps[2] = make(LayerMap)
	for i := 0; i < 16; i++ {
		network.LMaps[2][i] = make(map[int]float64)
	}

	network.Biases[0] = make([]float64, 16)
	network.Biases[1] = make([]float64, 16)
	network.Biases[2] = make([]float64, 10)

	// lMap0
	for i := 0; i < numOfPixels; i++ {
		for j := 0; j < 16; j++ {
			(network.LMaps[0])[i][j] = 1 - 2*rand.Float64()
		}
	}

	// lMap1
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			(network.LMaps[1])[i][j] = 1 - 2*rand.Float64()
		}
	}

	// lMap2
	for i := 0; i < 16; i++ {
		for j := 0; j < 10; j++ {
			(network.LMaps[2])[i][j] = 1 - 2*rand.Float64()
		}
	}

	// bias1
	for i := 0; i < 16; i++ {
		// network.Biases[0][i] = float64(0)
		network.Biases[0][i] = 1 - 2*rand.Float64()
	}

	// bias2
	for i := 0; i < 16; i++ {
		// network.Biases[1][i] = float64(0)
		network.Biases[1][i] = 1 - 2*rand.Float64()
	}

	// bias3
	for i := 0; i < 10; i++ {
		// network.Biases[2][i] = float64(0)
		network.Biases[2][i] = 1 - 2*rand.Float64()
	}

	return network
}

func getImage(archive TrainingImageFiles, fileNum int) []byte { // return the pixels of the specific image
	return archive.Pixels[fileNum*numOfPixels : (fileNum+1)*numOfPixels]
}

func showPicture(archive TrainingImageFiles, fileNum int) {
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

func parseImageFile(file *os.File) TrainingImageFiles {
	var testFile TrainingImageFiles
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

func parseLabelFile(file *os.File) TrainingLabelFiles {
	var labelFile TrainingLabelFiles
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
	return 1 / (1 + math.Exp(-1*x))
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
