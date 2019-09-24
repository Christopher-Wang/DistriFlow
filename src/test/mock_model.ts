import {Tensor, Variable} from '@tensorflow/tfjs';
import {DistributedClientModel} from '../client';
import {DistributedServerModel} from '../server';

export class MockModel implements DistributedServerModel, DistributedClientModel {
	isDistributedClientModel = true;
	isDistributedServerModel = true;
	inputShape = [1];
	outputShape = [1];
	vars: Variable[];
	version: string;

	constructor(vars: Variable[]) {
		this.vars = vars;
	}

	async setup() {}

	async save() {
		this.version = new Date().getTime().toString();
	}

	fit(x: Tensor, y: Tensor) {
		return this.vars;
	}

	update(gradients: Tensor[]) {}

	setVars(vars: Tensor[]) {
		for (let i = 0; i < this.vars.length; i++) {
			this.vars[i].assign(vars[i]);
		}
	}

	getVars(): Tensor[] {
		return this.vars;
	}

	predict(x: Tensor) {
		return x;
	}

	evaluate(x: Tensor, y: Tensor) {
		return [0];
	}
}
