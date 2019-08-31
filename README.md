# This is a NPM package for distributed machine learning on the browser
The goal of this package is to develop a lightweight tool for distributed training of tensorflow.js. The goal of this package is to have extendible framework to implement different ways for tensorflow.js models to be distributedly trained. 

Due to the volatile nature of webpages, I will not be considering model parallelism which requires synchronization between the distributed layers and will have many different failure cases. Instead, I will focus on data parallelism. 

For now, the primary goals is to implement synchronous batch gradient descent and asynchronous stochastic gradient descent, with a third stretch goal of implementing federated learning.

## DistriServer
```
+ model: DistriModel
+ dataset: DistriDataset
+ workerIDs: Set<ClientID>
```
The DistriServer is a simple lightweight websocket server. It is responsible for handling messages between itself and the DistriWorkers and maintaining state/application logic. 

## DistriWorker
```
+ clientID: UUID
+ model: DistriModel
```
The DistriWorker is a webworker to handle the computations in a non blocking way. It is responsible for computing the gradient on a minibatch.

## DistriModel
```
+ model: tf.LayersModel
+ modelID: UUID
+ maximumStaleness: Object
+ savedGradient: tf.grads (aggregated gradients local)
```
The DistriModel is a class that wraps a tf.layers model to be distributed and trained. It is responsible for implementing the logic to handle incoming gradients. 


## DistriDataset
```
+ epochIDs: List<UUID>
+ microBatchIDs: List<UUID>
+ queuedMicroBatch: int (index of MicroBatches)
+ microBatches: List<MicroBatchID>
+ incompletedMicroBatches: Set<MicroBatchID>
+ QueuedEpochs: int (index of Epoch)
+ Epochs: List<EpochID>
+ CompletedEpochs: Set<EpochID>
+ Dataset: tf.Dataset
```
The DistriDataset is a wrapper class for a tf.Dataset to be distributed. 