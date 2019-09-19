import '@tensorflow/tfjs-node';

import '../../src/server/fetch_polyfill';

import * as express from 'express';
import * as http from 'http';
import * as path from 'path';
import * as socketIO from 'socket.io';

import {SocketAPI} from '../../src/server/comm';
import {ModelDB} from '../../src/server/model_db';
import {MnistTransferLearningModel} from './mnist_transfer_learning_model';

const app = express();
const server = http.createServer(app);
const io = socketIO(server);
const dataDir = path.resolve(process.argv[2]);
const modelDB = new ModelDB(dataDir, parseInt(process.argv[3], 10));
const FIT_CONFIG = {
  batchSize: 1
};
const socketAPI = new SocketAPI(modelDB, FIT_CONFIG, io, true);

async function main() {
  await modelDB.setup(new MnistTransferLearningModel());
  await socketAPI.setup();
  await server.listen(3000);

  console.log('listening on 3000');
}

main();
