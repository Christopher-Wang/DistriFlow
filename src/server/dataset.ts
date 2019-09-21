import * as tf from '@tensorflow/tfjs';
import * as _ from 'underscore';
import { Batch, DEFAULT_DATASET_HYPERPARAMS, DistributedDatasetConfig, serializeVar, DataMsg, PreprocessCallback, UploadCallback } from '../common';

export class DistributedDataset{
    x: tf.Tensor;
    y: tf.Tensor;
    epoch: number;
    epochs: number;
    batchSize: number;
    batchIndex: number;
    batches: number;
    incompleteBatches: Set<number>;
    batchQueue: Iterator<number>;
    smallLastBatch: boolean;
    xBatchShape: number[];
    yBatchShape: number[];
    preprocessCallbacks: PreprocessCallback[];
    verbose: boolean;

    constructor(x: tf.Tensor, y: tf.Tensor, config: DistributedDatasetConfig){
        this.x = x;
        this.y = y;

        this.epochs = config.epochs || DEFAULT_DATASET_HYPERPARAMS.epochs;
        this.batchSize = config.batchSize || DEFAULT_DATASET_HYPERPARAMS.batchSize;
        this.smallLastBatch = config.smallLastBatch || DEFAULT_DATASET_HYPERPARAMS.smallLastBatch;

        this.epoch = 0;
        this.batches = Math.ceil(this.x.shape[0]/this.batchSize);
        this.incompleteBatches = new Set(_.range(this.batches));
        this.batchQueue = this.incompleteBatches.values();

        this.xBatchShape = _.toArray(this.x.shape);
        this.yBatchShape = _.toArray(this.y.shape);

        this.xBatchShape[0] = this.batchSize;
        this.yBatchShape[0] = this.batchSize;

        this.preprocessCallbacks = [];
    }

    public completeBatch(batch): boolean{
        return this.incompleteBatches.delete(batch);
    }

    public next(): { value: Batch, done: boolean }{
        if(this.incompleteBatches.size == 0){
            this.epoch++;
            if (this.epoch >= this.epochs) {
                return {value: undefined, done: true};
            }
            this.incompleteBatches = new Set(_.range(this.batches));
            this.batchQueue = this.incompleteBatches.values();
        }
       let {value, done} = this.batchQueue.next();
       if(done){
            this.batchQueue = this.incompleteBatches.values();
            ({value, done} = this.batchQueue.next());
       }
       let batch = this.getBatch(value);
       batch = this.preprocess(batch);
       return {
           value: batch,
           done: false
       };
    }

    private getBatch(batch: number): Batch{
        let xBatchIndex = new Array(this.x.rank).fill(0);
        let yBatchIndex = new Array(this.y.rank).fill(0);

        xBatchIndex[0] = this.batchSize * batch;
        yBatchIndex[0] = this.batchSize * batch;

        let x = this.x.slice(xBatchIndex, this.xBatchShape);
        let y = this.y.slice(yBatchIndex, this.yBatchShape);

        return {
            batch: batch,
            epoch: this.epoch,
            x: x,
            y: y
        }
    }

    private preprocess(batch: Batch): Batch{
        for (let preprocessCallback of this.preprocessCallbacks){
            batch = preprocessCallback(batch);
        }
        return batch;
    }

    addPreprocessCallback(callback: PreprocessCallback) {
        this.preprocessCallbacks.push(callback);
    }
}

export async function batchToDataMSG(batch: Batch): Promise<DataMsg>{
    let dataMSG = {
        batch: batch.batch,
        epoch: batch.epoch,
        x: await serializeVar(batch.x),
        y: await serializeVar(batch.y)
    };
    tf.dispose(batch.x);
    tf.dispose(batch.y);
    return dataMSG;
}