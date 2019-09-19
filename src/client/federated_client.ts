import * as tf from '@tensorflow/tfjs';

import { AbstractClient} from "./abstract_client";
import { addRows, sliceWithEmptyTensors } from "./utils";
import { Events, DownloadMsg, SerializedVariable, serializeVars, UploadMsg, DEFAULT_CLIENT_HYPERPARAMS } from "../common";

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