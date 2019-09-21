/**
 * Distributed Learning Client library.
 *
 * Example usage with a tf.Model:
 * ```js
 * const model = await tf.loadModel('a-model.json');
 * const client = new Client('http://server.com', model);
 * await client.setup();
 * await client.DistributedUpdate(data.X, data.y);
 * ```
 * The server->client synchronisation happens transparently whenever the server
 * broadcasts weights.
 * The client->server syncs happen periodically after enough `DistributedUpdate`
 * calls occur.
 */