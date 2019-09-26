import {tensor1d, tensor4d} from '@tensorflow/tfjs';
import {readFileSync} from 'fs';
import * as path from 'path';
import { DistributedDataset } from '../../src/server/dataset';
import { DEFAULT_DATASET_HYPERPARAMS, serializeVars, SerializedVariable, serializeVar, deserializeVars, DistributedTfModel } from '../../src';

const TRAIN_DATA = {
	imgs: path.resolve(__dirname, 'data/train-images-idx3-ubyte'),
	labels: path.resolve(__dirname, 'data/train-labels-idx1-ubyte')
};

const TEST_DATA = {
	imgs: path.resolve(__dirname, 'data/t10k-images-idx3-ubyte'),
	labels: path.resolve(__dirname, 'data/t10k-labels-idx1-ubyte')
};

function sliceIntoOwnBuffer(arr: Buffer): ArrayBuffer {
	return arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength);
}

function loadMnistFormat(imgsPath: string, labelsPath: string) {
	const imgsBytes = sliceIntoOwnBuffer(readFileSync(imgsPath).swap32());
	const labelsBytes = sliceIntoOwnBuffer(readFileSync(labelsPath).swap32());
	const imgsI32View = new Int32Array(imgsBytes);
	const labelsI32View = new Int32Array(labelsBytes);

	if (imgsI32View[0] !== 0x00000803) {
		throw new Error(
		'Training images file has invalid magic number 0x00000803 !== ' +
		imgsI32View[0].toString(16));
	}
	if (labelsI32View[0] !== 0x00000801) {
		throw new Error(
		'Training labels file has invalid magic number 0x00000801 !== ' +
		labelsI32View[0].toString(16));
	}

	const numItems = imgsI32View[1];
	const numRows = imgsI32View[2];
	const numCols = imgsI32View[3];
	const imgData = new Uint8Array(imgsBytes, 16, numItems * numRows * numCols);

	const imgsTensor =
	tensor4d(imgData, [numItems, numRows, numCols, 1], 'float32');

	if (labelsI32View[1] !== numItems) {
		throw new Error(`${numItems} images but ${labelsI32View[1]} labels`);
	}

	const labelsData = new Uint8Array(labelsBytes, 8, numItems);
	const labelsTensor = tensor1d(labelsData, 'int32');

	return {imgs: imgsTensor, labels: labelsTensor};
}

export function loadMnist() {
	return {
		train: loadMnistFormat(TRAIN_DATA.imgs, TRAIN_DATA.labels),
		val: loadMnistFormat(TEST_DATA.imgs, TEST_DATA.labels)
	};
}

export function loadDataset(): DistributedDataset{
	let mnist = loadMnist();
	let mnist_train = mnist['train'];
	let mnist_val = mnist['val'];
	// TODO : add preprocessing and validation
	// let x = batch.value.x;
	// let y = tf.oneHot(batch.value.y, 10);
	let labels = tf.oneHot(mnist_train['labels'], 10);
	return new DistributedDataset(mnist_train['imgs'], labels, DEFAULT_DATASET_HYPERPARAMS);
}



import * as tf from '@tensorflow/tfjs';
async function main() {

	// let batch = dist_dataset.next();
	// let model = createDenseModel();
	// let dist_model = new DistributedTfModel(model, {});

	


	// await dist_model.fetchInitial();
	// for(let i = 0; i < 10000; i++){
	// 	let grads = dist_model.fit(x, y);
	// 	let serializedGrads = await serializeVars(grads);
	// 	dist_model.update(deserializeVars(serializedGrads));
	// 	console.log(dist_model.evaluate(x, y));
	// }

	// const {value, grads} = tf.variableGrads(() => tf.losses.softmaxCrossEntropy(model.predictOnBatch(x), y).mean());
	// let gradList = Object.keys(grads).map( function(value, key){ return grads[value] });
	// let p = await serializeVars(gradList);
	// let q = deserializeVars(p);
	// console.log(q)
	
	//let sgrads = serializeVars(grads)

	// while(!batch.done){
	// 	if(Math.random() > -0.5){
	// 		console.log(`Batch: ${batch.value.batch} Epoch: ${batch.value.epoch}`);
	// 		dist_dataset.completeBatch(batch.value.batch);
	// 	}	
	// 	batch = dist_dataset.next();
	// }
}
// @ts-ignore
main();