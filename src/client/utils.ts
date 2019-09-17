import * as tf from '@tensorflow/tfjs';

const YEAR_IN_MS = 365 * 24 * 60 * 60 * 1000;

export async function fromEvent<T>(
    emitter: SocketIOClient.Socket, eventName: string,
    timeout: number): Promise<T> {
  return new Promise((resolve, reject) => {
           const rejectTimer = setTimeout(
               () => reject(`${eventName} event timed out`), timeout);
           const listener = (evtArgs: T) => {
             emitter.removeListener(eventName, listener);
             clearTimeout(rejectTimer);

             resolve(evtArgs);
           };
           emitter.on(eventName, listener);
         }) as Promise<T>;
}

// TODO: remove once tfjs >= 0.12.5 is released
export function concatWithEmptyTensors(a: tf.Tensor, b: tf.Tensor) {
	if (a.shape[0] === 0) {
		return b.clone();
	} else if (b.shape[0] === 0) {
		return a.clone();
	} else {
		return a.concat(b);
	}
}

export function sliceWithEmptyTensors(a: tf.Tensor, begin: number, size?: number) {
	if (begin >= a.shape[0]) {
		return tf.tensor([], [0].concat(a.shape.slice(1)));
	} else {
		return a.slice(begin, size);
	}
}

export function addRows(existing: tf.Tensor, newEls: tf.Tensor, unitShape: number[]) {
	if (tf.util.arraysEqual(newEls.shape, unitShape)) {
		return tf.tidy(() => concatWithEmptyTensors(existing, tf.expandDims(newEls)));
	} else {  // batch dimension
		tf.util.assertShapesMatch(newEls.shape.slice(1), unitShape);
		return tf.tidy(() => concatWithEmptyTensors(existing, newEls));
	}
}

export function getCookie(name: string) {
	if (typeof document === 'undefined') {
		return null;
	}
	const v = document.cookie.match('(^|;) ?' + name + '=([^;]*)(;|$)');
	return v ? v[2] : null;
}

export function setCookie(name: string, value: string) {
	if (typeof document === 'undefined') {
		return;
	}
	const d = new Date();
	d.setTime(d.getTime() + YEAR_IN_MS);
	document.cookie = name + '=' + value + ';path=/;expires=' + d.toUTCString();
}
