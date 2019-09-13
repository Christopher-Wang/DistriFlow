tf = require('@tensorflow/tfjs');
utils = require('./utils.js')
class DistriDataset{

	constructor(dataArray, numEpochs, minibatchSize){
		this.dataArray = dataArray;		
		this.numEpochs = numEpochs;
		this.minibatchSize = minibatchSize;
		this.epochIDs = [...utils.generateUUIDs(this.numEpochs)];
		this.minibatchIDs = [...utils.generateUUIDs(Math.ceil(numEpochs / minibatchSize))];
		this.incompleteMinibatches = new Set(this.minibatchIDs);
		this.minibatchQueue = this.incompleteMinibatches.values();
		this.queuedEpochs = 0;	
	}

	next(){
		if(incompleteMinibatches.size == 0){
			this.queuedEpochs++;
			if(this.queuedEpochs >= this.numEpochs){
				return {value: undefined, done: true };
			}
			this.incompleteMinibatches = new Set(this.minibatchIDs);
			this.minibatchQueue = this.incompleteMinibatches.values();
		}
					
		minibatchID = this.minibatchQueue.next();
		next = {
			value: {
				'minibatchID': minibatchID,
				'minibatch': this.dataArray[minibatchID],
				'epochID': this.queuedEpochs
			},
			done: false
		};
		return this.next;
	}

	completeMinibatch(minibatchID){
		return incompleteMinibatches.delete(minibatchID);
	}

	static async createDataset(dataset, numEpochs, minibatchSize){
		/*
			dataArray requires that the entire dataset fits in main memory. Further investigation into how to 
			maintain state using datastreams with Tensorflow.js is required
		*/
		const dataBatches = dataset.batch(minibatchSize)
		const dataArray = await (await dataBatches.iterator()).toArray();
		return new DistriDataset(dataArray, numEpochs, minibatchSize)
	}
}
test = async () => {
	const a = tf.data.array([1, 2, 3, 4, 5, 6, 7, 8]);
	d = await DistriDataset.createDataset(a, 10, 3);
	q =  new Set([1, 2, 3])
	console.log(q.delete(1))
	console.log({"done":false})
}

test();
