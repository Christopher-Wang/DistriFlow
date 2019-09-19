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
    //console.log(loadMnist()['train']['imgs'].shape);
    // await server.listen(3000);
    
    let model = await tf.loadLayersModel(mnistTransferLearningModelURL);
    let distmodel = new DistributedServerInMemoryModel(model, {})
    let fedserver = new FederatedServer(io, distmodel, {verbose: true});
    await fedserver.setup()
    await io.listen(80);
}

main();

