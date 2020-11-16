package main

import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"

	tf "github.com/tensorflow/tensorflow/tree/master/tensorflow/go"
)

var (
	graphFile  = "/model/tensorflow_inception_graph.pg"
	labelsFile = "/model/imagenet_comp_graph_label_strings.txt"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: imgrecognition <img_url>")
	}

	fmt.Printf("url: %s", os.Args[1])

	resp, err := http.Get(os.Args[1])

	if err != nil {
		log.Fatalf("unable to get image: %v", err)
	}

	defer resp.Body.Close()

	modelGraph, labels, err := loafGraphAndLabels()
	if err != nil {
		log.Fatalf("unable to load graph and labels: %v", err)
	}

}

func normalizeImage(body io.ReadCloser) (*tf.Tensor, error) {
	t, err := tf.NewTensor()
	if err != nil {
		return nil, nil
	}
}

func loafGraphAndLabels() (*tf.Graph, []string, error) {
	model, err := ioutil.ReadFile(graphFile)

	if err != nil {
		return nil, nil, err
	}

	g := tf.NewGraph()
	if err = g.Import(model, ""); err != nil {
		return nil, nil, err
	}

	f, err := os.Open(labelsFile)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	var labels []string
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	return g, labels, nil
}
