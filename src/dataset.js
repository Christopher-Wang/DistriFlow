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
		//Get the next minibatch from the queue
		minibatch = this.minibatchQueue.next();
		if(minibatch.done){
			//Check if there are no more incomplete minibatches for this epoch
			if(this.incompleteMinibatches == 0){
				this.queuedEpochs++;
				//Check if last epoch
				if(this.queuedEpochs >= this.numEpochs){
					return {value: undefined, done: true };
				}
				else{
					//Reset all minibatches
					this.incompleteMinibatches = new Set(this.minibatchIDs);
				}
			}
			//Reset minibatchQueue
			this.minibatchQueue = this.incompleteMinibatches.values();
			minibatch = this.minibatchQueue.next();
		}
		return {value: this.dataArray[minibatch], done: false }
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
	for (q  d){
		console.log(q);
	}
}

test();
