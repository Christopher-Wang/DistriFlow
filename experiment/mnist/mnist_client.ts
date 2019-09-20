import * as tf from '@tensorflow/tfjs';
import * as express from 'express';
import * as http from 'http';
import * as socketIO from 'socket.io';


import {DistributedServerInMemoryModel, FederatedServer, FederatedClient} from '../../src'
import { loadMnist } from './mnist_data';

const app = express();
const server = http.createServer(app);
const io = socketIO(server);
// tslint:disable-next-line:max-line-length
const mnistTransferLearningModelURL = 'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json';

async function main() {
    let model = await tf.loadLayersModel(mnistTransferLearningModelURL);
    let fedclient = new FederatedClient('http://localhost:80', model, {});
    await fedclient.setup()
}

main();

