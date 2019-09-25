import * as tf from '@tensorflow/tfjs';
// tslint:disable-next-line:max-line-length
import {LayerVariable, Tensor, Variable} from '@tensorflow/tfjs';

export type VarList = Array<Tensor|Variable|LayerVariable>;

export type SerializedVariable = {
	dtype: tf.DataType,
	shape: number[],
	data: ArrayBuffer
};

export const dtypeToTypedArrayCtor: {[index: string]:any} = {
	'float32': Float32Array,
	'int32': Int32Array,
	'bool': Uint8Array
};

export const lossesMap: {[index: string]: LossOrMetricFn} = {
	absoluteDifference : tf.losses.absoluteDifference,
	computeWeightedLoss: tf.losses.computeWeightedLoss,
	// TODO Investigate cosineDistance function signature as a loss function
	// cosineDistance: tf.losses.cosineDistance,
	hingeLoss: tf.losses.hingeLoss,
	huberLoss: tf.losses.huberLoss,
	logLoss: tf.losses.logLoss,
	meanSquaredError: tf.losses.meanSquaredError,
	sigmoidCrossEntropy: tf.losses.sigmoidCrossEntropy,
	softmaxCrossEntropy: tf.losses.softmaxCrossEntropy,
}

export async function serializeVar(variable: tf.Tensor): Promise<SerializedVariable> {
	const data = await variable.data();
	// small TypedArrays are views into a larger buffer
	const copy = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
	return {dtype: variable.dtype, shape: variable.shape.slice(), data: copy};
}

export async function serializeVars(vars: VarList) {
	const varsP: Array<Promise<SerializedVariable>> = [];
	vars.forEach((value, key) => {
		// tslint:disable-next-line:no-any
		const lv = (value as any);
		if (lv.write != null) {
			varsP.push(serializeVar(lv.read()));
		} else {
			varsP.push(serializeVar(lv));
		}
	});
	return Promise.all(varsP);
}

export function stackSerialized(vars: SerializedVariable[][]) {
	const updateCount = vars.length;
	const weightCount = vars[0].length;
	const stackedVars = [];

	for (let wt = 0; wt < weightCount; wt++) {
		const singleVar = vars[0][wt];
		const byteLength = singleVar.data.byteLength;
		const stackedVar = new Uint8Array(byteLength * updateCount);
		for (let up = 0; up < updateCount; up++) {
			const update = vars[up][wt].data;
			stackedVar.set(new Uint8Array(update), up * byteLength);
		}

		stackedVars.push({
			dtype: singleVar.dtype,
			shape: [updateCount].concat(singleVar.shape),
			data: stackedVar.buffer.slice(stackedVar.byteOffset, stackedVar.byteOffset + stackedVar.byteLength)
		});
	}

	return stackedVars;
}

export function deserializeVar(serialized: SerializedVariable): tf.Tensor {
	const array = serializedToArray(serialized);
	return tf.tensor(array, serialized.shape, serialized.dtype);
}

export function deserializeVars(vars: SerializedVariable[]) {
	return vars.map(deserializeVar);
}

export function serializedToArray(serialized: SerializedVariable) {
	const {dtype, shape, data: dataBuffer} = serialized;
	let data;
	// Because socket.io will deserialise JS ArrayBuffers into Nodejs Buffers
	if (dataBuffer instanceof ArrayBuffer) {
		data = dataBuffer;
		// tslint:disable-next-line no-any
	} else if ((dataBuffer as any) instanceof Buffer) {
		// tslint:disable-next-line no-any
		const dataAsBuffer = dataBuffer as any as Buffer;
		data = dataAsBuffer.buffer.slice(dataAsBuffer.byteOffset, dataAsBuffer.byteOffset + dataAsBuffer.byteLength);
	}
	const numel = shape.reduce((x, y) => x * y, 1);
	const ctor = dtypeToTypedArrayCtor[dtype];
	return new ctor(data, 0, numel);
}

export type LossOrMetricFn = (yTrue: Tensor, yPred: Tensor) => Tensor;

export type TfModelCallback = () => Promise<tf.LayersModel>;

export type AsyncTfModel = string|tf.LayersModel|TfModelCallback;

export type VersionCallback = (oldVersion: string, newVersion: string) => void;

export type UploadCallback = (msg: UploadMsg) => void;

export type PreprocessCallback = (batch: Batch) => Batch;

export enum Events {
	Download = 'downloadVars',
	Upload = 'uploadVars',
}

export type ModelMsg = {
	version: string,
	vars: SerializedVariable[]
};

export type GradientMsg = {
	version: string,
	vars: SerializedVariable[]
};

export type DataMsg = {
	batch: number,
	epoch: number,
	x: SerializedVariable,
	y: SerializedVariable
};

export type Batch = {
	batch: number,
	epoch: number,
	x: tf.Tensor,
	y: tf.Tensor
};

export type UploadMsg = {
	clientId: string,
	gradients?: GradientMsg,
	batch?: number,
	metrics?: number[]
};

export type DownloadMsg = {
	model: ModelMsg,
	hyperparams: ClientHyperparams,
	data?: DataMsg
};

export type DistributedCompileArgs = {
	loss?: string,
	learningRate?: number,
	metrics?: string[]
};

export type DistributedDatasetConfig = {
	epochs: number,
	batchSize: number,
	smallLastBatch?: boolean
}

export type ClientHyperparams = {
	batchSize?: number,          
	learningRate?: number,       
	epochs?: number,             
	examplesPerUpdate?: number
};

export type ServerHyperparams = {
	aggregation?: string,
	minUpdatesPerVersion?: number
};

export const DEFAULT_CLIENT_HYPERPARAMS: ClientHyperparams = {
	examplesPerUpdate: 5,
	learningRate: 0.001,
	batchSize: 32,
	epochs: 5
};

export const DEFAULT_SERVER_HYPERPARAMS: ServerHyperparams = {
	aggregation: 'mean',
	minUpdatesPerVersion: 20
};

export const DEFAULT_DATASET_HYPERPARAMS: DistributedDatasetConfig = {
	batchSize: 32,
	epochs: 5,
	smallLastBatch: false
};

export const DEFAULT_DISTRIBUTED_COMPILE_ARGS: DistributedCompileArgs = {
	loss: 'meanSquaredError',
	learningRate: 0.001,
	metrics: ['accuracy']
}

// tslint:disable-next-line:no-any
function override(defaults: any, choices: any) {
	// tslint:disable-next-line:no-any
	const result: any = {};
	for (const key in defaults) {
		result[key] = (choices || {})[key] || defaults[key];
	}
	for (const key in (choices || {})) {
		if (!(key in defaults)) {
			throw new Error(`Unrecognized key "${key}"`);
		}
	}
	return result;
}

export function clientHyperparams(hps?: ClientHyperparams): ClientHyperparams {
	try {
		return override(DEFAULT_CLIENT_HYPERPARAMS, hps);
	} catch (err) {
		throw new Error(`Error setting clientHyperparams: ${err.message}`);
	}
}

export function serverHyperparams(hps?: ServerHyperparams): ServerHyperparams {
	try {
		return override(DEFAULT_SERVER_HYPERPARAMS, hps);
	} catch (err) {
		throw new Error(`Error setting serverHyperparams: ${err.message}`);
	}
}

export async function fetchModel(asyncModel: AsyncTfModel): Promise<tf.LayersModel> {
	if (typeof asyncModel === 'string') {
		return await tf.loadLayersModel(asyncModel);
	} else if (typeof asyncModel === 'function') {
		return await asyncModel();
	} else {
		return asyncModel as tf.LayersModel;
	}
}