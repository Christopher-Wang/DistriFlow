import "jasmine";
import * as tf from '@tensorflow/tfjs';
import {test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line:max-line-length
import {deserializeVar, serializeVar, stackSerialized} from '../common';

describe('serialization', () => {
  it('converts back and forth to SerializedVar', async () => {
    const floatTensor = tf.tensor3d([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]]);
    const boolTensor = tf.tensor1d([true, false], 'bool');
    const intTensor = tf.tensor2d([[1, 2], [3, 4]], [2, 2], 'int32');

    const floatSerial = await serializeVar(floatTensor);
    const boolSerial = await serializeVar(boolTensor);
    const intSerial = await serializeVar(intTensor);
    const floatTensor2 = deserializeVar(floatSerial);
    const boolTensor2 = deserializeVar(boolSerial);
    const intTensor2 = deserializeVar(intSerial);
    test_util.expectArraysClose(floatTensor as any, floatTensor2 as any);
    test_util.expectArraysClose(boolTensor as any, boolTensor2 as any);
    test_util.expectArraysClose(intTensor as any, intTensor2 as any);
  });

  it('can stack lists of serialized variables', async () => {
    const floatTensor1 = tf.tensor3d([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const floatTensor2 = tf.tensor3d([[[0, 3], [0, 3]], [[0, 3], [0, 3]]]);
    const floatTensor3 = tf.tensor3d([[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]]);

    const intTensor1 = tf.tensor2d([[1, 2], [3, 4]], [2, 2], 'int32');
    const intTensor2 = tf.tensor2d([[5, 4], [3, 2]], [2, 2], 'int32');
    const intTensor3 = tf.tensor2d([[0, 0], [0, 0]], [2, 2], 'int32');

    const vars = [
      [await serializeVar(floatTensor1), await serializeVar(intTensor1)],
      [await serializeVar(floatTensor2), await serializeVar(intTensor2)],
      [await serializeVar(floatTensor3), await serializeVar(intTensor3)]
    ];

    const stack = stackSerialized(vars);

    const floatStack = deserializeVar(stack[0]);
    const intStack = deserializeVar(stack[1]);

    expect(floatStack.dtype).toBe('float32');
    expect(intStack.dtype).toBe('int32');

    test_util.expectArraysClose(floatStack.shape, [3, 2, 2, 2]);
    test_util.expectArraysClose(intStack.shape, [3, 2, 2]);
  });
});
