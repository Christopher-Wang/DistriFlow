import {tensor1d, tensor4d} from '@tensorflow/tfjs';
import {readFileSync} from 'fs';
import * as path from 'path';
import { DistributedDataset } from '../../src/server/dataset';
import { DEFAULT_DATASET_HYPERPARAMS } from '../../src';

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

async function main() {
	let mnist = loadMnist();
	let mnist_train = mnist['train'];
	let mnist_val = mnist['val'];
	let dist_dataset = new DistributedDataset(mnist_train['imgs'], mnist_train['labels'], DEFAULT_DATASET_HYPERPARAMS);
	let batch = dist_dataset.next();
	let i = 0;
	while(!batch.done){
		if(Math.random() > -0.5){
			console.log(`Batch: ${batch.value.batch} Epoch: ${batch.value.epoch}`);
			dist_dataset.completeBatch(batch.value.batch);
			i++;
		}	
		batch = dist_dataset.next();
	}
	console.log(i);
}

main();