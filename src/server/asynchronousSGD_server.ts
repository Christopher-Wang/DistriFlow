import * as tf from '@tensorflow/tfjs';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import * as io from 'socket.io';

import { AbstractServer, DistributedServerModel, DistributedServerConfig, isDistributedServerModel, DistributedServerTfModel } from ".";
import { AsyncTfModel, Events, UploadMsg, stackSerialized, deserializeVars } from "../common";

export class AsynchronousSGDServer extends AbstractServer {
    constructor(server: http.Server|https.Server|io.Server, 
        model: AsyncTfModel|DistributedServerModel, 
        config: DistributedServerConfig) {

        let ioServer = server;
        if (server instanceof http.Server || server instanceof https.Server) {
            ioServer = io(server);
        }

        let asyncSGDModel = model;
        if (!isDistributedServerModel(model)) {
            const defaultDir = path.resolve(`${process.cwd()}/saved-models`);
            const modelDir = config.modelDir || defaultDir;
            const compileConfig = config.modelCompileArgs || {};
            asyncSGDModel = new DistributedServerTfModel(modelDir, model, compileConfig);
        }

        if (!config.verbose) {
            config.verbose = (!!process.env.VERBOSE);
        }

        super(ioServer as io.Server, asyncSGDModel as DistributedServerModel, config);
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
            if (msg.model.version === this.model.version && !this.updating) {
                this.log(`new update from ${msg.clientId}`);
                this.updates.push(msg.model.vars);
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
            const newWeights = tf.tidy(() => {
            const stacked = stackSerialized(this.updates);
            const updates = deserializeVars(stacked);
            if (aggregation === 'mean') {
                return updates.map(update => update.mean(0));
            } else {
                throw new Error(`unsupported aggregation ${aggregation}`);
            }
            });
            this.model.setVars(newWeights);
            tf.dispose(newWeights);
        });

        this.model.save();
        this.downloadMsg = await this.computeDownloadMsg();
        this.updates = [];
        this.numUpdates = 0;
        this.updating = false;
        this.performCallbacks(oldVersion);
    }
}