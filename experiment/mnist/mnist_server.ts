import * as tf from '@tensorflow/tfjs';
import * as express from 'express';
import * as http from 'http';
import * as socketIO from 'socket.io';


import {DistributedServerInMemoryModel, FederatedServer, AsynchronousSGDServer} from '../../src'
import { loadMnist, loadDataset } from './mnist_data';

const app = express();
const server = http.createServer(app);
const io = socketIO(server);
// tslint:disable-next-line:max-line-length
const mnistTransferLearningModelURL = 'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json';

function createDenseModel() {
	const model = tf.sequential();
	model.add(tf.layers.flatten({inputShape: [28, 28, 1]}));
	model.add(tf.layers.dense({units: 10, activation: 'relu'}));
	model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
	return model;
}

async function main() {
    let dataset = loadDataset();
    //let model = await tf.loadLayersModel(mnistTransferLearningModelURL);
    let model = createDenseModel();
    let distmodel = new DistributedServerInMemoryModel(model, {})
    let fedserver = new AsynchronousSGDServer(io, distmodel, dataset, {verbose: true});
    fedserver.onUpload((msg) => {
        fedserver.log(`loss: ${msg.metrics[0]} accuracy:${msg.metrics[0]}`);
    })
    await fedserver.setup()
    await io.listen(80);
}

main();

