import * as tf from '@tensorflow/tfjs';

import { AbstractClient} from "./abstract_client";
import { addRows, sliceWithEmptyTensors } from "./utils";
import { Events, DownloadMsg, SerializedVariable, serializeVars, UploadMsg, DEFAULT_CLIENT_HYPERPARAMS, deserializeVar } from "../common";
import { timingSafeEqual } from 'crypto';

export class AsynchronousSGDClient extends AbstractClient{

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

		await this.time('Download weights from server', async () => {
			this.msg = await this.connectTo(this.server);

			this.setVars(this.msg.model.vars);
			const newVersion = this.modelVersion();
			this.versionUpdateCounts[newVersion] = 0;
			this.versionCallbacks.forEach(cb => cb(null, newVersion));
			this.DistributedUpdate();
		});
		

		this.socket.on(Events.Download, (msg: DownloadMsg) => {
			const oldVersion = this.modelVersion();
			const newVersion = msg.model.version;
			this.msg = msg;
			this.setVars(msg.model.vars);
			this.versionUpdateCounts[newVersion] = 0;
			this.versionCallbacks.forEach(cb => cb(oldVersion, newVersion));
			this.DistributedUpdate();
		});
	}

	public async DistributedUpdate(): Promise<void> {
		// save original ID (in case it changes during training/serialization)
		const modelVersion = this.modelVersion();
		this.x = deserializeVar(this.msg.data.x);
		this.y = deserializeVar(this.msg.data.y);

		// optionally compute evaluation metrics for them
		let metrics = null;
		if (this.sendMetrics) {
			metrics = this.model.evaluate(this.x, this.y);
		}

		// fit the model for the specified # of steps
		await this.time('Fit model', async () => {
			try {
				this.grads = this.model.fit(this.x, this.y);
			} catch (err) {
				console.error(err);
				throw err;
			}
		});
		
		//Serialize and dispose the gradients
		let gradients: SerializedVariable[] = await serializeVars(this.grads);
		// TODO : Investigate gradient memory leak, this is for the test
		//tf.dispose(this.grads);

		// upload the updates to the server
		const uploadMsg: UploadMsg = {
			batch: this.msg.data.batch,
			gradients: {version: modelVersion, vars: gradients},
			clientId: this.clientId,
		};

		if (this.sendMetrics) {
			uploadMsg.metrics = metrics;
		}

		await this.time('Upload weights to server', async () => {
			await this.uploadVars(uploadMsg);
		});
	}
}