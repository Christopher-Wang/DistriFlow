import * as tf from '@tensorflow/tfjs';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import * as io from 'socket.io';

import { AbstractServer, DistributedServerModel, DistributedServerConfig, isDistributedServerModel, DistributedServerTfModel, DistributedDataset, batchToDataMSG } from ".";
import { AsyncTfModel, Events, UploadMsg, stackSerialized, deserializeVars, DownloadMsg, serializeVars, DataMsg, SerializedVariable, GradientMsg } from "../common";

/**
 * The server aggregates model weight updates from clients and publishes new
 * versions of the model periodically to all clients.
 */
export class AsynchronousSGDServer extends AbstractServer {
    dataset: DistributedDataset;


    constructor(server: http.Server|https.Server|io.Server, 
        model: AsyncTfModel|DistributedServerModel, 
        config: DistributedServerConfig,
        dataset: DistributedDataset) {
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

        this.dataset = dataset;
    }

    async setup() {
        await this.time('setting up model', async () => {
            await this.model.setup();
        });

        this.server.on('connection', async (socket) => {
            this.numClients++;
            this.log(`connection: ${this.numClients} clients`);

            socket.on('disconnect', () => {
                this.numClients--;
                this.log(`disconnection: ${this.numClients} clients`);
            });

            let batch = this.dataset.next();
            if (batch.done){ return; }
            this.downloadMsg = await this.computeDownloadMsg(await batchToDataMSG(batch.value));
            await this.performCallbacks();
            this.log(`epoch: ${batch.value.epoch} batch: ${batch.value.batch}`);
            socket.emit(Events.Download, this.downloadMsg);

            socket.on(Events.Upload, async (msg: UploadMsg, ack) => {
                ack(true);
                this.log(`new update from ${msg.clientId}`);
                this.numUpdates++;

                await this.time('upload callbacks', async () => {
                    this.uploadCallbacks.forEach(c => c(msg));
                });

                this.dataset.completeBatch(msg.batch);
                await this.updateModel(msg.gradients);

                let batch = this.dataset.next();
                if (batch.done){ return; }
                this.downloadMsg = await this.computeDownloadMsg(await batchToDataMSG(batch.value));
                await this.performCallbacks();
                this.log(`epoch: ${batch.value.epoch} batch: ${batch.value.batch}`);
                this.server.sockets.emit(Events.Download, this.downloadMsg);
            });
        });
    }
    
    protected async computeDownloadMsg(data?: DataMsg): Promise<DownloadMsg> {
        return {
            model: {
                vars: await serializeVars(this.model.getVars()),
                version: this.model.version,
            },
            hyperparams: this.clientHyperparams,
            data: data
        };
    }

    private async updateModel(grads?: GradientMsg) {
        this.updating = true;
        const oldVersion = this.model.version;
        let gradsList = deserializeVars(grads.vars)

        await this.time('computing new weights', async () => {
            this.model.updateVars(gradsList);
        });

        tf.dispose(gradsList);
        this.model.save();
        this.updating = false;
        this.performCallbacks(oldVersion);
    }
}