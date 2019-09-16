import * as tf from '@tensorflow/tfjs';
import { DistributedModel, DistributedTfModel, DistributedDynamicModel } from "../common/models";
import * as fs from 'fs';
import {promisify} from 'util';
import { AsyncTfModel, DistributedCompileArgs, dtypeToTypedArrayCtor } from '../common/utils';


const readdir = promisify(fs.readdir);
const exists = promisify(fs.exists);
const mkdir = promisify(fs.mkdir);
const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);
const symlink = promisify(fs.symlink);
const unlink = promisify(fs.unlink);
const readlink = promisify(fs.readlink);

async function forceSymlink(src: string, dest: string) {
	try {
		await symlink(src, dest, 'dir');
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code !== 'EEXIST') {
			throw err;
		}
		const existingLink = await readlink(dest);
		if (src !== existingLink) {
			await unlink(dest);
			await symlink(src, dest, 'dir');
		}
	}
}

/**
 * DistributedServerModel describes the interface that models passed to `Server`
 * must implement.
 *
 * See the DistributedModel documentation in src/common/index.ts for more details.
 */
export interface DistributedServerModel extends DistributedModel {
	isDistributedServerModel: boolean;
	version: string;

	/**
	 * Initialize the model
	 */
	setup(): Promise<void>;

	/**
	 * Save the current model and update `version`.
	 */
	save(): Promise<void>;
}

/**
* Type guard for Distributed server models.
*
* @param model any object
*/
// tslint:disable-next-line:no-any
export function isDistributedServerModel(model: any): model is DistributedServerModel {
	return model && model.isDistributedServerModel;
}

export class DistributedServerInMemoryModel extends DistributedTfModel implements DistributedServerModel {
	isDistributedServerModel = true;
	version: string;

	async setup() {
		const isBrowser = tf.ENV.get('IS_BROWSER');
		tf.ENV.set('IS_BROWSER', true);  // TODO: remove me in tfjs 0.12.5
		await this.fetchInitial();
		tf.ENV.set('IS_BROWSER', isBrowser);
		await this.save();
	}

	async save() {
		this.version = new Date().getTime().toString();
	}
}


/**
 * Specific version of DistributedServerModel that wraps a `tf.Model`,
 * an async function returning a `tf.Model`, or a string that can be passed to
 * `tf.loadLayersModel`.
 *
 * Stores models as subdirectories of `saveDir`. Different model versions are
 * identified by timestamps.
 */
export class DistributedServerTfModel extends DistributedTfModel implements DistributedServerModel {
	isDistributedServerModel = true;
	saveDir: string;
	version: string;

	constructor(
		saveDir: string, initialModel?: AsyncTfModel,
		config?: DistributedCompileArgs) {
	super(initialModel, config);
	this.saveDir = saveDir;
	}

	async setup() {
		if (!(await exists(this.saveDir))) {
			await mkdir(this.saveDir);
		}
		const last = await this.last();
		if (last) {
			await this.load(last);
		} else {
			tf.ENV.set('IS_BROWSER', true);  // TODO: remove me in tfjs 0.12.5
			await this.fetchInitial();
			tf.ENV.set('IS_BROWSER', false);
			await this.save();
		}
	}

	async list() {
		const models = await readdir(this.saveDir);
		const idx = models.indexOf('current');
		if (idx >= 0) {
			models.splice(idx);
		}
		models.sort();
		return models;
	}

	async last() {
		const models = await this.list();
		if (models.length) {
			return models[models.length - 1];
		} else {
			return null;
		}
	}

	async save() {
		const version = new Date().getTime().toString();
		this.version = version;
		const url = `file://${this.saveDir}/${version}`;
		await this.model.save(url);
		await forceSymlink(`${this.saveDir}/${version}`, `${this.saveDir}/current`);
	}

	async load(version: string) {
		const url = `file://${this.saveDir}/${version}/model.json`;
		this.version = version;
		this.model = await tf.loadLayersModel(url);
		this.model.compile(this.compileConfig);
		await forceSymlink(`${this.saveDir}/${version}`, `${this.saveDir}/current`);
	}
}

export class DistributedServerDynamicModel extends DistributedDynamicModel implements DistributedServerModel {
	saveDir: string;
	version = '';
	isDistributedServerModel = true;

	constructor(args: {
		saveDir: string, vars: tf.Variable[];
		predict: (inputs: tf.Tensor) => tf.Tensor;
		loss: (labels: tf.Tensor, preds: tf.Tensor) => tf.Scalar;
		optimizer: tf.Optimizer;
		inputShape: number[];
		outputShape: number[];
	}) {
		super(args);
		this.saveDir = args.saveDir;
		this.save();
	}

	async setup() {
		if (!(await exists(this.saveDir))) {
			await mkdir(this.saveDir);
		}
		const last = await this.last();
		if (last) {
			await this.load(last);
		} else {
			await this.save();
		}
	}

	async list() {
		const models = await readdir(this.saveDir);
		models.sort();
		return models;
	}

	async last() {
		const models = await this.list();
		if (models.length) {
			return models[models.length - 1];
		} else {
			return null;
		}
	}

	async save() {
		const version = new Date().getTime().toString();
		this.version = version;
		const path = `${this.saveDir}/${version}/`;
		await mkdir(path);
		const jsonPath = `${path}/meta.json`;
		const binPath = `${path}/data.bin`;
		const {data, json} = await flatSerialize(this.vars);
		await writeFile(jsonPath, JSON.stringify(json));
		await writeFile(binPath, data);
	}

	async load(version: string) {
		const path = `${this.saveDir}/${version}/`;
		const jsonPath = `${path}/meta.json`;
		const binPath = `${path}/data.bin`;
		const json = JSON.parse(await readFile(jsonPath, {encoding: 'utf8'}));
		const data = await readFile(binPath);
		return flatDeserialize({data, json});
	}
}

export type FlatVars = {
	data: Uint8Array,
	json: {
		meta: Array<{shape: number[], dtype: 'string' | 'float32' | 'int32' | 'bool'| 'complex64'}>,
		byteOffsets: number[]
	}
};

function unview(a: ArrayBuffer|ArrayBufferView) {
	if (ArrayBuffer.isView(a)) {
		return a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength);
	} else {
		return a;
	}
}

export async function flatSerialize(tensors: tf.Tensor[]): Promise<FlatVars> {
	const meta = tensors.map(({shape, dtype}) => ({shape, dtype}));

	const datas = await Promise.all(tensors.map(t => t.data().then(unview)));

	const totBytes = datas.map(({byteLength}) => byteLength).reduce((x, y) => x + y);

	const dataArr = new Uint8Array(totBytes);

	let cursor = 0;
	const byteOffsets = [];

	for (const buf of datas) {
		dataArr.set(new Uint8Array(buf), cursor);
		byteOffsets.push(cursor);
		cursor += buf.byteLength;
	}

	return {data: dataArr, json: {meta, byteOffsets}};
}

export function flatDeserialize({data, json: {meta, byteOffsets}}: FlatVars) {
	const numels = meta.map(({shape}) => shape.reduce((x, y) => x * y, 1));

	const tensors = meta.map(({shape, dtype}, i) => {
		const ctor = dtypeToTypedArrayCtor[dtype];
		const arr = new ctor(data.buffer, byteOffsets[i], numels[i]);
		return tf.tensor(arr, shape, dtype);
	});

	return tensors;
}