import * as tf from '@tensorflow/tfjs';
// tslint:disable-next-line:max-line-length
import {ModelCompileArgs, Tensor} from '@tensorflow/tfjs';
import {AsyncTfModel, fetchModel, DistributedFitConfig, DistributedCompileArgs, DEFAULT_CLIENT_HYPERPARAMS} from './utils';

export interface DistributedModel {
    /**
     * Trains the model distributedly to better predict the given targets.
     *
     * @param x `tf.Tensor` of training input data.
     * @param y `tf.Tensor` of training target data.
     * @param config optional fit configuration.
     *
     * @return A `Promise` resolved when training is done.
     */
    fit(x: Tensor, y: Tensor, config?: DistributedFitConfig): Promise<void>;
  
    /**
     * Makes predictions on input data.
     *
     * @param x `tf.Tensor` of input data.
     *
     * @return model ouputs
     */
    predict(x: Tensor): Tensor;
  
    /**
     * Evaluates performance on data.
     *
     * @param x `tf.Tensor` of input data.
     * @param y `tf.Tensor` of target data.
     *
     * @return An array of evaluation metrics.
     */
    evaluate(x: Tensor, y: Tensor): number[];
  
    /**
     * Gets the model's variables.
     *
     * @return A list of `tf.Variable`s or LayerVariables representing the model's
     * trainable weights.
     */
    getVars(): Tensor[];
  
    /**
     * Sets the model's variables to given values.
     *
     * @param vals An array of `tf.Tensor`s representing updated model weights
     */
    setVars(vals: Tensor[]): void;
  
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
    compileConfig: ModelCompileArgs;
    private _initialModel: AsyncTfModel;

    constructor(initialModel?: AsyncTfModel, config?: DistributedCompileArgs) {
        this._initialModel = initialModel;
        this.compileConfig = {
        loss: config.loss || 'categoricalCrossentropy',
        metrics: config.metrics || ['accuracy'],
        optimizer: 'sgd'
        };
    }

    async fetchInitial() {
        if (this._initialModel) {
        this.model = await fetchModel(this._initialModel);
        this.model.compile(this.compileConfig);
        } else {
        throw new Error('no initial model provided!');
        }
    }

    async fit(x: Tensor, y: Tensor, config?: DistributedFitConfig) {
        if (config.learningRate) {
        (this.model.optimizer as tf.SGDOptimizer)
            .setLearningRate(config.learningRate);
        }
        await this.model.fit(x, y, {
        epochs: config.epochs || DEFAULT_CLIENT_HYPERPARAMS.epochs,
        batchSize: config.batchSize || DEFAULT_CLIENT_HYPERPARAMS.batchSize
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
        this.inputShape = args.inputShape;
        this.outputShape = args.outputShape;
    }

    async setup() {
        return Promise.resolve();
    }

    async fit(x: tf.Tensor, y: tf.Tensor, config?: DistributedFitConfig):
        Promise<void> {
        if (config.learningRate) {
        this.optimizer.setLearningRate(config.learningRate);
        }
        const epochs = (config && config.epochs) || 1;
        for (let i = 0; i < epochs; i++) {
        const ret = this.optimizer.minimize(() => this.loss(y, this.predict(x)));
        tf.dispose(ret);
        }
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