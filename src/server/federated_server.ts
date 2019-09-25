import * as tf from '@tensorflow/tfjs';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import * as io from 'socket.io';

import { AbstractServer, DistributedServerModel, DistributedServerConfig, isDistributedServerModel, DistributedServerTfModel } from ".";
import { AsyncTfModel, Events, UploadMsg, stackSerialized, deserializeVars } from "../common";

/**
 * Distributed Learning Server library.
 *
 * Example usage with a tf.Model:
 * ```js
 * const model = await tf.loadModel('file:///a/model.json');
 * const webServer = http.createServer();
 * const fedServer = new Server(webServer, model);
 * fedServer.setup().then(() => {
 *  webServer.listen(80);
 * });
 * ```
 *
 * The server aggregates model weight updates from clients and publishes new
 * versions of the model periodically to all clients.
 */
export class FederatedServer extends AbstractServer {
    constructor(server: http.Server|https.Server|io.Server, 
        model: AsyncTfModel|DistributedServerModel, 
        config: DistributedServerConfig) {
        // Setup server
        let ioServer = server;
        if (server instanceof http.Server || server instanceof https.Server) {
            ioServer = io(server);
        }

        // Setup model
        let fedModel = model;
        if (!isDistributedServerModel(model)) {
            const defaultDir = path.resolve(`${process.cwd()}/saved-models`);
            const modelDir = config.modelDir || defaultDir;
            const compileConfig = config.modelCompileArgs || {};
            fedModel = new DistributedServerTfModel(modelDir, model, compileConfig);
        }

        if (!config.verbose) {
            config.verbose = (!!process.env.VERBOSE);
        }

        super(ioServer as io.Server, fedModel as DistributedServerModel, config);
    }

    async setup() {
        await this.time('setting up model', async () => {
            await this.model.setup();
        });

        this.downloadMsg = await this.computeDownloadMsg();
        await this.performCallbacks();

        this.server.on('connection', (socket) => {
            this.numClients++;
            this.log(`connection: ${this.numClients} clients`);

            socket.on('disconnect', () => {
                this.numClients--;
                this.log(`disconnection: ${this.numClients} clients`);
            });

            socket.emit(Events.Download, this.downloadMsg);

            socket.on(Events.Upload, async (msg: UploadMsg, ack) => {
            ack(true);
            if (msg.gradients.version === this.model.version && !this.updating) {
                this.log(`new update from ${msg.clientId}`);
                this.updates.push(msg.gradients.vars);
                this.numUpdates++;
                await this.time('upload callbacks', async () => {
                    this.uploadCallbacks.forEach(c => c(msg));
                });
                if (this.shouldUpdate()) {
                await this.updateModel();
                    this.server.sockets.emit(Events.Download, this.downloadMsg);
                }
            }
            });
        });
    }

    private shouldUpdate(): boolean {
        const numUpdates = this.numUpdates;
        return (numUpdates >= this.serverHyperparams.minUpdatesPerVersion);
    }

    private async updateModel() {
        this.updating = true;
        const oldVersion = this.model.version;
        const aggregation = this.serverHyperparams.aggregation;

        await this.time('computing new weights', async () => {
            const newgrads = tf.tidy(() => {
                    const stacked = stackSerialized(this.updates);
                    const updates = deserializeVars(stacked);
                    if (aggregation === 'mean') {
                        return updates.map(update => update.mean(0));
                    } else {
                        throw new Error(`unsupported aggregation ${aggregation}`);
                    }
                });
            this.model.update(newgrads);
            tf.dispose(newgrads);
        });

        this.model.save();
        this.downloadMsg = await this.computeDownloadMsg();
        this.updates = [];
        this.numUpdates = 0;
        this.updating = false;
        this.performCallbacks(oldVersion);
    }
}