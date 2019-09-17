import * as tf from '@tensorflow/tfjs';
import * as socketProxy from 'socket.io-client';
import * as uuid from 'uuid/v4';

// tslint:disable-next-line:max-line-length
import { ClientHyperparams, DownloadMsg, SerializedVariable, VersionCallback, UploadCallback, Events, UploadMsg, serializeVars, DistributedCompileArgs, AsyncTfModel, DEFAULT_CLIENT_HYPERPARAMS, deserializeVar } from '../common/utils';
import { DistributedClientModel, isDistributedClientModel, DistributedClientTfModel } from './models';
import { getCookie, setCookie, addRows, sliceWithEmptyTensors, fromEvent } from './utils';

// tslint:disable-next-line:no-angle-bracket-type-assertion no-any
const socketio = (<any>socketProxy).default || socketProxy;
const CONNECTION_TIMEOUT = 10 * 1000;
const UPLOAD_TIMEOUT = 5 * 1000;
const COOKIE_NAME = 'Distributed-learner-uuid';

type CounterObj = {
	[key: string]: number
};

type SocketCallback = () => SocketIOClient.Socket;

export type DistributedClientConfig = {
	modelCompileConfig?: DistributedCompileArgs,
	hyperparams?: ClientHyperparams,
	verbose?: boolean,
	clientId?: string,
	sendMetrics?: boolean
};

export abstract class AbstractClient{
	protected msg: DownloadMsg;
	protected model: DistributedClientModel;
	protected socket: SocketIOClient.Socket;
	protected versionCallbacks: VersionCallback[];
	protected uploadCallbacks: UploadCallback[];
	protected versionUpdateCounts: CounterObj;
	protected server: string|SocketCallback;
	protected verbose: boolean;
	protected sendMetrics: boolean;
	hyperparams: ClientHyperparams;
	clientId: string;

	/**
	 * Construct a client API for Distributed learning that will push and pull
	 * `model` updates from the server.
	 * @param model - model to use with Distributed learning
	 */
	constructor(server: string|SocketCallback, model: DistributedClientModel|AsyncTfModel, config?: DistributedClientConfig) {
		this.server = server;
		if (isDistributedClientModel(model)) {
			this.model = model;
		} else {
			const compileConfig = (config || {}).modelCompileConfig || {};
			this.model = new DistributedClientTfModel(model, compileConfig);
		}
		this.uploadCallbacks = [];
		this.versionCallbacks = [(v1, v2) => {
			this.log(`Updated model: ${v1} -> ${v2}`);
		}];
		this.versionUpdateCounts = {};
		this.verbose = (config || {}).verbose;
		this.sendMetrics = (config || {}).sendMetrics;
		if ((config || {}).clientId) {
			this.clientId = config.clientId;
		} else if (getCookie(COOKIE_NAME)) {
			this.clientId = getCookie(COOKIE_NAME);
		} else {
			this.clientId = uuid();
			setCookie(COOKIE_NAME, this.clientId);
		}
		this.hyperparams = (config || {}).hyperparams || {};
	}

	/**
	 * @return The version of the model we're currently training
	 */
	public modelVersion(): string {
		return this.msg == null ? 'unsynced' : this.msg.model.version;
	}

	/**
	 * Register a new callback to be invoked whenever the server updates the model
	 * version.
	 *
	 * @param callback function to be called (w/ old and new version IDs)
	 */
	onNewVersion(callback: VersionCallback) {
		this.versionCallbacks.push(callback);
	}

	/**
	 * Register a new callback to be invoked whenever the client uploads a new set
	 * of weights.
	 *
	 * @param callback function to be called (w/ client's upload msg)
	 */
	onUpload(callback: UploadCallback) {
		this.uploadCallbacks.push(callback);
	}

	evaluate(x: tf.Tensor, y: tf.Tensor): number[] {
		return this.model.evaluate(x, y);
	}

	predict(x: tf.Tensor): tf.Tensor {
		return this.model.predict(x);
	}

	numUpdates(): number {
		let numTotal = 0;
		Object.keys(this.versionUpdateCounts).forEach(k => {
			numTotal += this.versionUpdateCounts[k];
		});
		return numTotal;
	}

	numVersions(): number {
		return Object.keys(this.versionUpdateCounts).length;
	}

	dispose(): void {
		this.socket.disconnect();
		this.log('Disconnected');
	}

	get inputShape(): number[] {
		return this.model.inputShape;
	}

	get outputShape(): number[] {
		return this.model.outputShape;
	}

	// tslint:disable-next-line:no-any
	protected log(...args: any[]) {
		if (this.verbose) {
			console.log('Distributed Client:', ...args);
		}
	}

	/**
	 * Upload the current values of the tracked variables to the server
	 * @return A promise that resolves when the server has recieved the variables
	 */
	protected async uploadVars(msg: UploadMsg): Promise<{}> {
		const prom = new Promise((resolve, reject) => {
			const rejectTimer =
				setTimeout(() => reject(`uploadVars timed out`), UPLOAD_TIMEOUT);
			this.socket.emit(Events.Upload, msg, () => {
			clearTimeout(rejectTimer);
			resolve();
			});
		});
		return prom;
	}

	protected setVars(newVars: SerializedVariable[]) {
		tf.tidy(() => {
			this.model.setVars(newVars.map(v => deserializeVar(v)));
		});
	}

	protected async connectTo(server: string|SocketCallback): Promise<DownloadMsg> {
	if (typeof server === 'string') {
		this.socket = socketio(server);
	} else {
		this.socket = server();
	}
	return fromEvent<DownloadMsg>(this.socket, Events.Download, CONNECTION_TIMEOUT);
	}

	protected async time(msg: string, action: () => Promise<void>) {
		const t1 = new Date().getTime();
		await action();
		const t2 = new Date().getTime();
		this.log(`${msg} took ${t2 - t1}ms`);
	}
}

/**
 * Distributed Learning Client library.
 *
 * Example usage with a tf.Model:
 * ```js
 * const model = await tf.loadModel('a-model.json');
 * const client = new Client('http://server.com', model);
 * await client.setup();
 * await client.DistributedUpdate(data.X, data.y);
 * ```
 * The server->client synchronisation happens transparently whenever the server
 * broadcasts weights.
 * The client->server syncs happen periodically after enough `DistributedUpdate`
 * calls occur.
 */
export class FederatedClient extends AbstractClient{
	private x: tf.Tensor;
	private y: tf.Tensor;

	/**
	 * Connect to a server, synchronise the variables to their initial values
	 * @param serverURL: The URL of the server
	 * @return A promise that resolves when the connection has been established
	 * and variables set to their inital values.
	 */
	public async setup(): Promise<void> {
		await this.time('Initial model setup', async () => {
			await this.model.setup();
		});
		this.x = tf.tensor([], [0].concat(this.model.inputShape));
		this.y = tf.tensor([], [0].concat(this.model.outputShape));
		await this.time('Download weights from server', async () => {
			this.msg = await this.connectTo(this.server);
		});
		this.setVars(this.msg.model.vars);
		const newVersion = this.modelVersion();
		this.versionUpdateCounts[newVersion] = 0;
		this.versionCallbacks.forEach(cb => cb(null, newVersion));

		this.socket.on(Events.Download, (msg: DownloadMsg) => {
			const oldVersion = this.modelVersion();
			const newVersion = msg.model.version;
			this.msg = msg;
			this.setVars(msg.model.vars);
			this.versionUpdateCounts[newVersion] = 0;
			this.versionCallbacks.forEach(cb => cb(oldVersion, newVersion));
		});
	}

	/**
	 * Train the model on the given examples, upload new weights to the server,
	 * then revert back to the original weights (so subsequent updates are
	 * relative to the same model).
	 *
	 * Note: this method will save copies of `x` and `y` when there
	 * are too few examples and only train/upload after reaching a
	 * configurable threshold (disposing of the copies afterwards).
	 *
	 * @param x Training inputs
	 * @param y Training labels
	 */
	public async DistributedUpdate(x: tf.Tensor, y: tf.Tensor): Promise<void> {
		// incorporate examples into our stored `x` and `y`
		const xNew = addRows(this.x, x, this.model.inputShape);
		const yNew = addRows(this.y, y, this.model.outputShape);
		tf.dispose([this.x, this.y]);
		this.x = xNew;
		this.y = yNew;

		// repeatedly, for as many iterations as we have batches of examples:
		const examplesPerUpdate = this.hyperparam('examplesPerUpdate');
		while (this.x.shape[0] >= examplesPerUpdate) {
			// save original ID (in case it changes during training/serialization)
			const modelVersion = this.modelVersion();

			// grab the right number of examples
			const xTrain = sliceWithEmptyTensors(this.x, 0, examplesPerUpdate);
			const yTrain = sliceWithEmptyTensors(this.y, 0, examplesPerUpdate);
			const fitConfig = {
				epochs: this.hyperparam('epochs'),
				batchSize: this.hyperparam('batchSize'),
				learningRate: this.hyperparam('learningRate')
			};

			// optionally compute evaluation metrics for them
			let metrics = null;
			if (this.sendMetrics) {
				metrics = this.model.evaluate(xTrain, yTrain);
			}

			// fit the model for the specified # of steps
			await this.time('Fit model', async () => {
				try {
					await this.model.fit(xTrain, yTrain, fitConfig);
				} catch (err) {
					console.error(err);
					throw err;
				}
			});

			// serialize, possibly adding noise
			const stdDev = this.hyperparam('weightNoiseStddev');
			let newVars: SerializedVariable[];
			if (stdDev) {
				const newTensors = tf.tidy(() => {
					return this.model.getVars().map(v => {
						return v.add(tf.randomNormal(v.shape, 0, stdDev));
					});
				});
				newVars = await serializeVars(newTensors);
				tf.dispose(newTensors);
			} else {
				newVars = await serializeVars(this.model.getVars());
			}

			// revert our model back to its original weights
			this.setVars(this.msg.model.vars);

			// upload the updates to the server
			const uploadMsg: UploadMsg = {
				model: {version: modelVersion, vars: newVars},
				clientId: this.clientId,
			};
			if (this.sendMetrics) {
				uploadMsg.metrics = metrics;
			}
			await this.time('Upload weights to server', async () => {
				await this.uploadVars(uploadMsg);
			});
			this.uploadCallbacks.forEach(cb => cb(uploadMsg));
			this.versionUpdateCounts[modelVersion] += 1;
			
			tf.dispose([xTrain, yTrain]);
			const xRest = sliceWithEmptyTensors(this.x, examplesPerUpdate);
			const yRest = sliceWithEmptyTensors(this.y, examplesPerUpdate);
			tf.dispose([this.x, this.y]);
			this.x = xRest;
			this.y = yRest;
		}
	}
	
	public numExamples(): number {
		return this.x.shape[0];
	}

	private hyperparam(key: 'batchSize'|'learningRate'|'epochs'|'examplesPerUpdate'|'weightNoiseStddev'): number {
		return (this.hyperparams[key] || this.msg.hyperparams[key] || DEFAULT_CLIENT_HYPERPARAMS[key]);
	}

	public numExamplesPerUpdate(): number {
		return this.hyperparam('examplesPerUpdate');
	}

	public numExamplesRemaining(): number {
		return this.numExamplesPerUpdate() - this.numExamples();
	}
}