/**
Convenience function for generating UUIDs
@param numUUIDs the number of UUIDs to generate
*/
function *generateUUIDs(numUUIDs) {
	for (let i = 0; i < numUUIDs; i++) {
		yield i;
	}
}

/**
Serializes a tf.tensor, tf.variable or tf.LayerVariable
@param variable The variable to be serialized
@return A promise of a SerializedVariable
*/
async function serializeVar(variable){
	const data = await variable.data();
	// small TypedArrays are views into a larger buffer
	const copy = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
	return {'dtype': variable.dtype, 'shape': variable.shape.slice(), 'data': copy};
}

/**
Serializes a variable list of tf.tensors, tf.variables or tf.LayerVariables
@param variables The variablelist to be serialized
@return A list of promises coomposed of SerializedVariables
*/
async function serializeVars(variables) {
  const varsP = [];
  variables.forEach((value, key) => {
    if (lv.write != null) {
      varsP.push(serializeVar(value.read()));
    } else {
      varsP.push(serializeVar(value));
    }
  });
  return Promise.all(varsP);
}


module.exports = {
	generateUUIDs: generateUUIDs
}