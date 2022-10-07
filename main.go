package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/moobu/moo/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	creds := insecure.NewCredentials()
	cc, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(creds))
	if err != nil {
		log.Fatal(err)
	}

	client := proto.NewInferenceClient(cc)

	f, err := os.Open("example/ankle-boot.jpg")
	if err != nil {
		log.Fatal(err)
	}

	b, err := io.ReadAll(f)
	if err != nil {
		log.Fatal(err)
	}
	in := proto.PredictRequest{
		Batch: []*proto.Input{
			{Body: b, Meta: nil},
		},
	}
	out, err := client.Predict(context.TODO(), &in)
	if err != nil {
		log.Fatal(err)
	}

	var v map[string]float32
	for _, output := range out.Batch {
		err = json.Unmarshal(output.Body, &v)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(v)
	}
}
