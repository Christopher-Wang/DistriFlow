/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import * as io from 'socket.io';
import {Server as IOServer} from 'socket.io';
// tslint:disable-next-line:max-line-length
import { ClientHyperparams, ServerHyperparams, DownloadMsg, SerializedVariable, VersionCallback, UploadCallback, clientHyperparams, serverHyperparams, Events, UploadMsg, serializeVars, stackSerialized, deserializeVars, DistributedCompileArgs, AsyncTfModel } from '../common/utils';
import { DistributedServerModel, isDistributedServerModel, DistributedServerTfModel } from './models';


export type DistributedServerConfig = {
    clientHyperparams?: ClientHyperparams,
    serverHyperparams?: ServerHyperparams,
    updatesPerVersion?: number,
    modelDir?: string,
    modelCompileArgs?: DistributedCompileArgs,
    verbose?: boolean
};

export class AbstractServer {
    model: DistributedServerModel;
    clientHyperparams: ClientHyperparams;
    serverHyperparams: ServerHyperparams;
    downloadMsg: DownloadMsg;
    server: IOServer;
    numClients = 0;
    numUpdates = 0;
    updates: SerializedVariable[][] = [];
    updating = false;
    versionCallbacks: VersionCallback[];
    uploadCallbacks: UploadCallback[];
    verbose: boolean;

    constructor(webServer: IOServer, model: DistributedServerModel, config: DistributedServerConfig) {
        // Setup server
        this.server = webServer;
        this.model = model;
        this.verbose = (!!config.verbose) || false;
        this.clientHyperparams = clientHyperparams(config.clientHyperparams || {});
        this.serverHyperparams = serverHyperparams(config.serverHyperparams || {});
        this.downloadMsg = null;
        this.uploadCallbacks = [];
        this.versionCallbacks = [(v1, v2) => {
            this.log(`updated model: ${v1} -> ${v2}`);
        }];
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

    protected async computeDownloadMsg(): Promise<DownloadMsg> {
        return {
            model: {
            vars: await serializeVars(this.model.getVars()),
            version: this.model.version,
            },
            hyperparams: this.clientHyperparams
        };
    }

    // tslint:disable-next-line:no-any
    protected log(...args: any[]) {
        if (this.verbose) {
            console.log('Distributed Server:', ...args);
        }
    }

    protected async time(msg: string, action: () => Promise<void>) {
        const t1 = new Date().getTime();
        await action();
        const t2 = new Date().getTime();
        this.log(`${msg} took ${t2 - t1}ms`);
    }

    protected async performCallbacks(oldVersion?: string) {
        await this.time('performing callbacks', async () => {
            this.versionCallbacks.forEach(c => c(oldVersion, this.model.version));
        });
    }
}


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