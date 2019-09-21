import * as tf from '@tensorflow/tfjs';

import { AbstractClient} from "./abstract_client";
import { addRows, sliceWithEmptyTensors } from "./utils";
import { Events, DownloadMsg, SerializedVariable, serializeVars, UploadMsg, DEFAULT_CLIENT_HYPERPARAMS } from "../common";

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

	public async DistributedUpdate(): Promise<void> {
        // TODO FINISH CLIENT
	}
}