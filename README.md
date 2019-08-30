# This is a NPM package for distributed machine learning on the browser
The goal of this package is to develop a lightweight tool to build, train and distribute the training of tensorflow.js models in two methods: synchronous machine learing and asyncronous machine learning. This project will create a websocket server that communicates model parameters, data and gradients between clients.

## DistriServer
Instantiates a websocket server
Instantiates a DistriModel
Instantiates a DistriDataset

## DistriWorker
Recieves a DistriModel
Recieves a DistriDataset 

## DistriModel
+ Tensorflow LayersModel
+ CompletedGradientID: UUID (corresponds to MicroBatchID)
+ SavedGradient: tf.grads (aggregated gradients)

+ update(): Tensorflow LayersModel

## DistriDataset
+ EpochID: UUID
+ MicroBatchID:UUID
+ QueuedMicroBatch: int (index of MicroBatches)
+ MicroBatches: List<MicroBatchID>
+ CompletedMicroBatches: Set<MicroBatchID>
+ QueuedEpochs: int (index of Epoch)
+ Epochs: List<EpochID>
+ CompletedEpochs: Set<EpochID>

+ next(): tf.Dataset