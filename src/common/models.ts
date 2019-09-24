import * as tf from '@tensorflow/tfjs';

// tslint:disable-next-line:max-line-length
import {Tensor} from '@tensorflow/tfjs';
import {AsyncTfModel, fetchModel, DistributedCompileArgs, DEFAULT_CLIENT_HYPERPARAMS, LossOrMetricFn, lossesMap, DEFAULT_DISTRIBUTED_COMPILE_ARGS} from './utils';

export interface DistributedModel {
    /**
     * Trains the model distributedly to better predict the given targets.
     *
     * @param x `tf.Tensor` of training input data.
     * @param y `tf.Tensor` of training target data.
     *
     * @return A list of `tf.Tensors`s that are the gradients of the current
     * model given an (x, y) batch
     */
    fit(x: tf.Tensor, y: tf.Tensor): tf.Tensor[];
  

    /**
     * Updates the model's variables given the gradients
     *
     * @param grads An array of `tf.Tensor`s representing the model's gradients
     */
    update(grads: tf.Tensor[]): void;


    /**
     * Makes predictions on input data.
     *
     * @param x `tf.Tensor` of input data.
     *
     * @return model ouputs
     */
    predict(x: tf.Tensor): tf.Tensor;
  
    /**
     * Evaluates performance on data.
     *
     * @param x `tf.Tensor` of input data.
     * @param y `tf.Tensor` of target data.
     *
     * @return An array of evaluation metrics.
     */
    evaluate(x: tf.Tensor, y: tf.Tensor): number[];
  
    /**
     * Gets the model's variables.
     *
     * @return A list of `tf.Variable`s or LayerVariables representing the model's
     * trainable weights.
     */
    getVars(): tf.Tensor[];
  
    /**
     * Sets the model's variables to given values.
     *
     * @param vals An array of `tf.Tensor`s representing updated model weights
     */
    setVars(vals: tf.Tensor[]): void;


    /**
     * Shape of model inputs (not including the batch dimension)
     */
    inputShape: number[];
  
    /**
     * Shape of model outputs (not including the batch dimension)
     */
    outputShape: number[];
}
  
export class DistributedTfModel implements DistributedModel {
    model: tf.LayersModel;
    learningRate: tf.Scalar;
    loss:  string;
    lossFunction: LossOrMetricFn;
    metrics: string[];
    optimizer: string;
    private _initialModel: AsyncTfModel;

    constructor(initialModel: AsyncTfModel, config?: DistributedCompileArgs) {
        this._initialModel = initialModel;
        this.loss = config.loss || DEFAULT_DISTRIBUTED_COMPILE_ARGS.loss;
        this.lossFunction = lossesMap[this.loss];
        this.metrics = config.metrics || DEFAULT_DISTRIBUTED_COMPILE_ARGS.metrics;
        this.optimizer = 'sgd';
        this.learningRate = tf.scalar(config.learningRate || DEFAULT_DISTRIBUTED_COMPILE_ARGS.learningRate);
    }

    async fetchInitial() {
        this.model = await fetchModel(this._initialModel);
        // TODO : Investigate model compiling in a distributed setting
        this.model.compile({
            loss: this.loss,
            metrics: this.metrics,
            optimizer: this.optimizer,
        });
    }

    predict(x: Tensor) {
        return this.model.predict(x) as Tensor;
    }

    evaluate(x: Tensor, y: Tensor) {
        return tf.tidy(() => {
        const results = this.model.evaluate(x, y);
        if (results instanceof Array) {
            return results.map(r => r.dataSync()[0]);
        } else {
            return [results.dataSync()[0]];
        }
        });
    }

    getVars(): tf.Tensor[] {
        return this.model.trainableWeights.map((v) => v.read());
    }

    // TODO: throw friendly error if passed variable of wrong shape?
    setVars(vals: tf.Tensor[]) {
        for (let i = 0; i < vals.length; i++) {
            this.model.trainableWeights[i].write(vals[i]);
        }
    }

    update(grads: tf.Tensor[]) {
        let vars = this.model.trainableWeights.map((v) => v);
        vars.forEach((v, i) => {
            v.write(tf.tidy(() => {
                return v.read().sub(this.learningRate.mul(grads[i]))
            }));
        });
    }

    fit(x: tf.Tensor, y: tf.Tensor): tf.Tensor[]{
        // @ts-ignore
        const {value, grads} = tf.variableGrads(() => tf.losses.softmaxCrossEntropy(this.model.predictOnBatch(x), y).mean());
        let gradients = Object.keys(grads).map( function(value, key){ return grads[value] });
        return gradients;
    }

    get inputShape() {
        return this.model.inputLayers[0].batchInputShape.slice(1);
    }

    get outputShape() {
        return (this.model.outputShape as number[]).slice(1);
    }
}
  
export class DistributedDynamicModel implements DistributedModel {
    isDistributedClientModel = true;
    version: string;
    vars: tf.Variable[];
    predict: (inputs: tf.Tensor) => tf.Tensor;
    loss: (labels: tf.Tensor, preds: tf.Tensor) => tf.Scalar;
    optimizer: tf.SGDOptimizer;
    learningRate: tf.Scalar;
    inputShape: number[];
    outputShape: number[];

    constructor(args: {
        vars: tf.Variable[]; predict: (inputs: tf.Tensor) => tf.Tensor;
        loss: (labels: tf.Tensor, preds: tf.Tensor) => tf.Scalar;
        inputShape: number[];
        outputShape: number[];
    }) {
        this.vars = args.vars;
        this.predict = args.predict;
        this.loss = args.loss;
        this.optimizer = tf.train.sgd(DEFAULT_CLIENT_HYPERPARAMS.learningRate);
        this.learningRate = tf.scalar(DEFAULT_CLIENT_HYPERPARAMS.learningRate);
        this.inputShape = args.inputShape;
        this.outputShape = args.outputShape;
    }

    async setup() {
        return Promise.resolve();
    }

    update(grads: tf.Tensor[]) {
        this.vars.forEach((v, i) => {
            v.assign(tf.tidy(() => v.add(-this.learningRate.mul(grads[i]))));
        });
    }

    fit(x: tf.Tensor, y: tf.Tensor): tf.Tensor[]{
        const {value, grads} = tf.variableGrads(() => this.loss(y, this.predict(x)).mean());
        let gradList = Object.keys(grads).map( function(value, key){ return grads[value] });
        return gradList;
    }

    evaluate(x: tf.Tensor, y: tf.Tensor): number[] {
        return Array.prototype.slice.call(this.loss(y, this.predict(x)).dataSync());
    }

    getVars() {
        return this.vars;
    }

    setVars(vals: tf.Tensor[]) {
        this.vars.forEach((v, i) => {
            v.assign(vals[i]);
        });
    }
}