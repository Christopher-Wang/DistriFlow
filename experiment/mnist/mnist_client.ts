import * as tf from '@tensorflow/tfjs';
import * as express from 'express';
import * as http from 'http';
import * as socketIO from 'socket.io';


import { AsynchronousSGDClient, FederatedClient} from '../../src';

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
    //let model = await tf.loadLayersModel(mnistTransferLearningModelURL);

    let model = createDenseModel();
    let fedclient = new AsynchronousSGDClient('http://localhost:80', model, {sendMetrics: true});
    await fedclient.setup()
}

main();

