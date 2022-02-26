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
	showPicture(testFile, 1)
}

func showPicture(archive TrainingSetImageFiles, fileNum int) {
	if fileNum > int(archive.NumberOfImages) {
		panic("fileNum is to high")
	}

	pixels := archive.Pixels
	numOfPixels := int(archive.NumberOfRows * archive.NumberOfColumns)

	decide := func(b byte) rune {
		if b <= 256/2 {
			return ' '
		} else {
			return 'x'
		}
	}

	i := 0
	for i < numOfPixels {
		fmt.Printf("%c ", decide(pixels[i]))
		if i%int(archive.NumberOfColumns) == 0 && i != 0 {
			fmt.Printf("\n")
		}

		i++
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
