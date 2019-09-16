import {DistributedModel, DistributedTfModel} from '../common/models';

/**
 * Interface that DistributedClientModels must support. Essentially a wrapper
 * around DistributedModel, which is defined in src/common/index.ts.
 */
export interface DistributedClientModel extends DistributedModel {
	isDistributedClientModel: boolean;

	setup(): Promise<void>;
}

/**
 * Specific version of DistributedClientModel that wraps a `tf.Model`, async
 * function returning a `tf.Model`, or a string that can be passed to
 * `tf.loadModel`.
 */
export class DistributedClientTfModel extends DistributedTfModel implements DistributedClientModel {
	isDistributedClientModel = true;

	async setup() {
		await this.fetchInitial();
	}
}

/**
 * Type guard for DistributedClientModel.
 *
 * @param model any object
 */
// tslint:disable-next-line:no-any
export function isDistributedClientModel(model: any): model is DistributedClientModel {
	return model && model.isDistributedClientModel;
}
