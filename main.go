package main

import (
	"encoding/binary"
	"fmt"
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

func main() {
	file, err := os.Open("t10k-images.idx3-ubyte")
	check(err)
	defer file.Close()

	testFile := parseImageFile(file)
	for i := 0; i < 12; i++ {
		showPicture(testFile, i*11)
	}
}

func showPicture(archive TrainingSetImageFiles, fileNum int) {
	if fileNum > int(archive.NumberOfImages) {
		panic("fileNum is to high")
	}

	numOfPixels := int(archive.NumberOfRows * archive.NumberOfColumns)
	pixels := archive.Pixels[fileNum*numOfPixels : (fileNum+1)*numOfPixels]

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
		printByte(pixels[i])

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

	numOfPixels := testFile.NumberOfImages * testFile.NumberOfRows * testFile.NumberOfColumns
	pixels := make([]byte, numOfPixels)

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
