package main

import (
	"encoding/binary"
	"fmt"
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

type Network struct {
	LMaps  [3]LayerMap
	Biases [4]([]float64)
}

var numOfPixels int

// don't forget the bias

func main() {
	file, err := os.Open("t10k-images.idx3-ubyte")
	check(err)
	defer file.Close()

	testFile := parseImageFile(file)
	numOfPixels = int(testFile.NumberOfRows * testFile.NumberOfColumns)

	network := initNetwork()

	ratings := guessDigit(getImage(testFile, 1), network)

	fmt.Println(ratings)

	// for i := 0; i < 12; i++ {
	// 	showPicture(testFile, i*11)
	// }
}

func guessDigit(image []byte, network Network) [10]float64 {
	var layer1Nodes [16]float64
	var layer2Nodes [16]float64
	var layer3Nodes [10]float64

	// layer0 -> layer1
	for i := 0; i < 16; i++ {
		for j := 0; i < numOfPixels; j++ {
			layer1Nodes[i] += (network.LMaps[0])[j][i] * float64(image[j])
		}
	}

	// layer1 -> layer2
	for i := 0; i < 16; i++ {
		for j := 0; i < 16; j++ {
			layer2Nodes[i] += (network.LMaps[1])[j][i] * layer1Nodes[j]
		}
	}

	// layer2 -> layer3
	for i := 0; i < 10; i++ {
		for j := 0; i < 16; j++ {
			layer3Nodes[i] += (network.LMaps[2])[j][i] * layer2Nodes[j]
		}
	}

	return layer3Nodes
}

// initialize the maps between layers randomly
func initNetwork() Network {
	var network Network

	network.LMaps[0] = make(LayerMap)
	network.LMaps[1] = make(LayerMap)
	network.LMaps[2] = make(LayerMap)

	network.Biases[0] = make([]float64, numOfPixels)
	network.Biases[1] = make([]float64, 16)
	network.Biases[2] = make([]float64, 16)

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
	for i := 0; i < numOfPixels; i++ {
		network.Biases[0][i] = float64(0)
	}

	// bias1
	for i := 0; i < 16; i++ {
		network.Biases[1][i] = float64(0)
	}

	// bias2
	for i := 0; i < 16; i++ {
		network.Biases[2][i] = float64(0)
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

	n, err := file.Read(pixels)
	check(err)
	fmt.Printf("Read %v pixels from file\n", n)

	testFile.Pixels = pixels

	return testFile
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
