import "jasmine";
import * as tf from '@tensorflow/tfjs';
import {FederatedClient} from '../client';
import {FederatedServer} from '../server';
import * as fs from 'fs';
import * as http from 'http';
import * as rimraf from 'rimraf';
import {MockModel} from './mock_model';

const PORT = 3001;
const socketURL = `http://0.0.0.0:${PORT}`;
const initWeights = [tf.tensor([1, 1, 1, 1], [2, 2]), tf.tensor([1, 2, 3, 4], [1, 4])];
const initVersion = 'initial';

describe('Server-to-client API', () => {
	let dataDir: string;
	let server: FederatedServer;
	let client: FederatedClient;
	let clientVars: tf.Variable[];
	let serverVars: tf.Variable[];
	let httpServer: http.Server;

	beforeEach(async () => {
		// Set up model database with our initial weights
		dataDir = fs.mkdtempSync('/tmp/Distributed_test');

		clientVars = initWeights.map(t => tf.variable(tf.zerosLike(t)));
		serverVars = initWeights.map(t => tf.variable(t));
		const clientModel = new MockModel(clientVars);
		const serverModel = new MockModel(serverVars);
		serverModel.version = initVersion;

		// Set up the server exposing our upload/download API
		httpServer = http.createServer();
		await httpServer.listen(PORT);

		server = new FederatedServer(httpServer, serverModel, {
			modelDir: dataDir,
			serverHyperparams: {
			minUpdatesPerVersion: 2,
			},
			clientHyperparams: {examplesPerUpdate: 1}
		});
		await server.setup();

		client = new FederatedClient(socketURL, clientModel);
		await client.setup();
		});

		afterEach(async () => {
		rimraf.sync(dataDir);
		await httpServer.close();
		await client.dispose();
	});

	it('transmits model version on startup', async () => {
		expect(client.modelVersion()).toBe(initVersion);
	});

	it('transmits updates', async () => {
		expect(server.updates.length).toBe(0);

		clientVars[0].assign(tf.tensor([2, 2, 2, 2], [2, 2]));
		const dummyX = tf.tensor2d([[0]]);
		const dummyY = tf.tensor2d([[0]]);
		await client.DistributedUpdate(dummyX, dummyY);

		expect(server.updates.length).toBe(1);
	});

	it('triggers a download after enough uploads', async (done) => {
		client.onNewVersion((oldVersion, newVersion) => {
			expect(oldVersion).toBe(initVersion);
			expect(newVersion).not.toBe(initVersion);
			expect(newVersion).toBe(server.model.version);
			done();
		});

		const dummyX1 = tf.tensor2d([[0]]);  // 1 example
		const dummyY1 = tf.tensor2d([[0]]);
		const dummyX3 = tf.tensor2d([[0], [0], [0]]);  // 3 examples
		const dummyY3 = tf.tensor2d([[0], [0], [0]]);
		clientVars[0].assign(tf.tensor([2, 2, 2, 2], [2, 2]));
		clientVars[1].assign(tf.tensor([1, 2, 3, 4], [1, 4]));
		await client.DistributedUpdate(dummyX1, dummyY1);
		clientVars[0].assign(tf.tensor([1, 1, 1, 1], [2, 2]));
		clientVars[1].assign(tf.tensor([5, 4, 3, 1], [1, 4]));
		await client.DistributedUpdate(dummyX3, dummyY3);
		
	});
});
