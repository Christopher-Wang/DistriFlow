import * as tf from '@tensorflow/tfjs';
import * as socketProxy from 'socket.io-client';
import * as uuid from 'uuid/v4';

// tslint:disable-next-line:max-line-length
import { ClientHyperparams, DownloadMsg, SerializedVariable, VersionCallback, UploadCallback, Events, UploadMsg, DistributedCompileArgs, AsyncTfModel, deserializeVar } from '../common/utils';
import { DistributedClientModel, isDistributedClientModel, DistributedClientTfModel } from './models';
import { getCookie, setCookie, fromEvent } from './utils';

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
	protected grads: tf.Tensor[];
	protected x: tf.Tensor;
	protected y: tf.Tensor;
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
