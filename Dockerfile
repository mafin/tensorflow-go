FROM ctava/tfcgo

RUN mkdir -p /model && \
 curl -o /model/inception5h.zip https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && \
 unzip /model/inception5h.zip -d /model && rm /model/LICENSE

WORKDIR /go/src/imgrecognition
COPY . .
RUN go build

ENTRYPOINT ["/go/src/imgrecognition/imgrecognition"]
