import * as tf from '@tensorflow/tfjs';

import {Scalar, Tensor} from '@tensorflow/tfjs';
import {FederatedModel, ModelDict} from '../../src/index';

// https://github.com/tensorflow/tfjs-examples/tree/master/mnist-transfer-cnn
const mnistTransferLearningModelURL =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json';

export class MnistTransferLearningModel implements FederatedModel {
  async setup(): Promise<ModelDict> {
    const oriModel = await tf.loadLayersModel(mnistTransferLearningModelURL);
    const frozenLayers = oriModel.layers.slice(0, 10);

    frozenLayers.forEach(layer => layer.trainable = false);

    const headLayers = [tf.layers.dense({units: 10})];

    const model = tf.sequential({layers: frozenLayers.concat(headLayers)});
    const loss = (inputs: Tensor, labels: Tensor) => {
      const logits = model.predict(inputs) as Tensor;
      const losses = tf.losses.softmaxCrossEntropy(labels, logits);
      return losses.mean() as Scalar;
    };

    return {predict: model.predict, vars: model.trainableWeights, loss, model};
  }
}
