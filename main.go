package main

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"sort"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	graphFile  = "/model/tensorflow_inception_graph.pb"
	labelsFile = "/model/imagenet_comp_graph_label_strings.txt"
)

type Label struct {
	Label       string
	Probability float32
}

type Labels []Label

func (l Labels) Len() int           { return len(l) }
func (l Labels) Swap(i, j int)      { l[i], l[j] = l[j], l[i] }
func (l Labels) Less(i, j int) bool { return l[i].Probability > l[j].Probability }

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

	session, err := tf.NewSession(modelGraph, nil)
	if err != nil {
		log.Fatalf("unable to init session: %v", err)
	}
	defer session.Close()

	tensor, err := normalizeImage(resp.Body)
	if err != nil {
		log.Fatalf("unable to normalize image: %v", err)
	}

	result, err := session.Run(map[tf.Output]*tf.Tensor{
		modelGraph.Operation("input").Output(0): tensor,
	},
		[]tf.Output{
			modelGraph.Operation("output").Output(0),
		}, nil)
	if err != nil {
		log.Fatalf("unable to inference: %v", err)
	}

	topFiveLabels := getTopFiveLabels(labels, result[0].Value().([][]float32)[0])

	for _, l := range topFiveLabels {
		fmt.Printf("label: %s, probability: %.2f%%\n", l.Label, l.Probability*100)
	}
}

func getTopFiveLabels(labels []string, probabilities []float32) []Label {
	var results []Label

	for i, p := range probabilities {
		if i >= len(labels) {
			break
		}

		results = append(results, Label{
			Label:       labels[i],
			Probability: p,
		})
	}

	sort.Sort(Labels(results))

	return results[:5]
}

func normalizeImage(body io.ReadCloser) (*tf.Tensor, error) {
	var buf bytes.Buffer
	io.Copy(&buf, body)

	t, err := tf.NewTensor(buf.String())
	if err != nil {
		return nil, nil
	}

	graph, input, output, err := getNormalizedImage()
	if err != nil {
		return nil, err
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()

	normalized, err := session.Run(map[tf.Output]*tf.Tensor{
		input: t,
	},
		[]tf.Output{
			output,
		}, nil)
	if err != nil {
		return nil, err
	}

	return normalized[0], nil
}

func getNormalizedImage() (*tf.Graph, tf.Output, tf.Output, error) {
	s := op.NewScope()
	input := op.Placeholder(s, tf.String)
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))

	output := op.Sub(s,
		op.ResizeBilinear(s,
			op.ExpandDims(s,
				op.Cast(s, decode, tf.Float),
				op.Const(s.SubScope("make_batch"), int32(0))),
			op.Const(s.SubScope("size"), []int32{224, 224})),
		op.Const(s.SubScope("mean"), float32(117)))
	graph, err := s.Finalize()

	return graph, input, output, err
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
