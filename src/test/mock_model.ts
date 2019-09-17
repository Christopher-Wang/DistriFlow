import {Tensor, Variable} from '@tensorflow/tfjs';
import {DistributedClientModel} from '../client';
import {DistributedFitConfig} from '../common'
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

  async fit(x: Tensor, y: Tensor, config?: DistributedFitConfig) {}

  async setup() {}

  async save() {
    this.version = new Date().getTime().toString();
  }

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
