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

import {Server as IOServer} from 'socket.io';
// tslint:disable-next-line:max-line-length
import { ClientHyperparams, ServerHyperparams, DownloadMsg, SerializedVariable, VersionCallback, UploadCallback, clientHyperparams, serverHyperparams, serializeVars, DistributedCompileArgs} from '../common/utils';
import { DistributedServerModel} from './models';


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