import * as tf from "@tensorflow/tfjs";
import { Graph, graphFromPredicates, addInverseRelation } from "./graph";

function printParameterDistribution(model: tf.LayersModel) {
  model.layers.forEach((layer, index) => {
    if (layer.getWeights().length > 0) {
      console.log(`Layer ${index} (${layer.name}):`);

      const weights = layer.getWeights()[0].dataSync();
      const biases = layer.getWeights()[1].dataSync();

      console.log("  Weights:");
      printDistributionStats(weights);

      console.log("  Biases:");
      printDistributionStats(biases);
    }
  });
}

function printDistributionStats(values: any) {
  const mean = tf.mean(values).dataSync()[0];
  const std = tf.moments(values).variance.sqrt().dataSync()[0];
  const min = tf.min(values).dataSync()[0];
  const max = tf.max(values).dataSync()[0];

  console.log(`    Mean: ${mean.toFixed(6)}`);
  console.log(`    Std Dev: ${std.toFixed(6)}`);
  console.log(`    Min: ${min.toFixed(6)}`);
  console.log(`    Max: ${max.toFixed(6)}`);
}

function makeMLPPredicateModel(
  relation_vocab: number,
  node_vocab: number,
  hidden: number,
  n_layers: number = 1,
  l2_regularization: number = 0.01,
  dropout: number = 0,
  learning_rate: number = 3e-4,
): tf.LayersModel {
  const model = tf.sequential();

  for (var i = 0; i < n_layers; i++) {
    model.add(
      tf.layers.dense({
        units: hidden,
        activation: "relu",
        inputShape: i == 0 ? [relation_vocab + node_vocab] : [hidden],
        kernelRegularizer: tf.regularizers.l2({ l2: l2_regularization }),
        biasRegularizer: tf.regularizers.l2({ l2: l2_regularization }),
        kernelInitializer: tf.initializers.glorotNormal({ seed: 777 }),
        biasInitializer: tf.initializers.glorotNormal({ seed: 777 }),
      }),
    );
    model.add(tf.layers.dropout({ rate: dropout, seed: 777 }));
  }

  model.add(
    tf.layers.dense({
      units: node_vocab,
      kernelRegularizer: tf.regularizers.l2({ l2: l2_regularization }),
      biasRegularizer: tf.regularizers.l2({ l2: l2_regularization }),
      kernelInitializer: tf.initializers.glorotNormal({ seed: 777 }),
      biasInitializer: tf.initializers.glorotNormal({ seed: 777 }),
    }),
  );
  model.compile({
    optimizer: tf.train.adam(learning_rate),
    loss: tf.losses.softmaxCrossEntropy,
  });

  return model;
}

function lossOnIdx(
  model: tf.LayersModel,
  dataset: DataSet,
  idx: number,
): number {
  const x: tf.Tensor = dataset.xs.slice([idx, 0], [1, -1]);
  const y: tf.Tensor = dataset.ys.slice([idx, 0], [1, -1]);
  const yhat: tf.Tensor = (model.call(x, {}) as tf.Tensor[])[0];
  const result: number = tf.losses.softmaxCrossEntropy(y, yhat).dataSync()[0];

  console.log("======================");
  console.log("x");
  x.print();
  console.log("y");
  y.print();
  console.log("yhat");
  yhat.print();
  console.log({ result });

  return result;
}

// Training loop
async function trainModel(model: tf.LayersModel, dataset: TrainTestDataset) {
  dataset.train.xs.print();
  dataset.train.ys.print();
  printParameterDistribution(model);
  for (let epoch = 0; epoch < 1000; epoch++) {
    const history = await model.fit(dataset.train.xs, dataset.train.ys, {
      epochs: 1,
      batchSize: dataset.train.xs.shape[0],
      verbose: 0,
      validationData: [dataset.test.xs, dataset.test.ys],
    });
    console.log(
      `Epoch ${epoch + 1}: loss = ${history.history.loss[0]}, val_loss = ${history.history.val_loss[0]}`,
    );
  }

  for (var i = 0; i < dataset.train.xs.shape[0]; i++) {
    lossOnIdx(model, dataset.train, i);
  }
}

interface DataSet {
  xs: tf.Tensor;
  ys: tf.Tensor;
}

interface TrainTestDataset {
  train: DataSet;
  test: DataSet;
  split_idxs: boolean[];
}

function indexify<T>(values: Iterable<T>): Map<T, number> {
  const result: Map<T, number> = new Map();

  for (const value of values) {
    if (!result.has(value)) {
      result.set(value, result.size);
    }
  }

  return result;
}

/*
 * Converts a graph to onehot tensors
 * The xs are a concatenation of the 'relation' and 'from' node along feature dim
 * both encoded as onehot vectors
 *
 * @param {graph}: graph to be converted to tensors
 *
 * @returns {dataset}: tensor encodings for a task prediction 'to' node from
 * 	'relation' and 'from' node
 */
function graphToTFJSData(graph: Graph): DataSet {
  // TODO: this function is very close to just being 3 calls to
  // categoricalToOneHot(edges, feature_fn), it would be nice to find an elegant
  // way to do that
  const edge_types: Map<string, number> = indexify(
    Array.from(graph.edges).map((edge) => edge.relation),
  );
  const node_idxs: Map<string, number> = indexify(Array.from(graph.nodes));

  console.log(
    Array.from(graph.edges).map((edge) => edge_types.get(edge.relation)!),
  );

  const edge_tensor: tf.Tensor = tf.oneHot(
    tf
      .tensor(
        Array.from(graph.edges).map((edge) => edge_types.get(edge.relation)!),
      )
      .toInt(),
    edge_types.size,
  );
  const from_node_tensor: tf.Tensor = tf.oneHot(
    tf
      .tensor(
        Array.from(graph.edges).map((edge) => node_idxs.get(edge.fromNode)!),
      )
      .toInt(),
    node_idxs.size,
  );
  const to_node_tensor: tf.Tensor = tf.oneHot(
    tf
      .tensor(
        Array.from(graph.edges).map((edge) => node_idxs.get(edge.toNode)!),
      )
      .toInt(),
    node_idxs.size,
  );

  return {
    xs: tf.concat([edge_tensor, from_node_tensor], 1),
    ys: to_node_tensor,
  };
}

async function maskDataset(
  dataset: DataSet,
  mask: boolean[],
): Promise<DataSet> {
  return {
    xs: await tf.booleanMaskAsync(dataset.xs, mask, 0),
    ys: await tf.booleanMaskAsync(dataset.ys, mask, 0),
  };
}

async function trainTestSplit(
  dataset: DataSet,
  ratio: number,
): Promise<TrainTestDataset> {
  const n_samples: number = dataset.xs.shape[0];
  const inTrainSet: boolean[] = Array(n_samples).map(
    (_) => Math.random() < ratio,
  );
  const inTestSet: boolean[] = inTrainSet.map((x) => !x);

  return {
    train: await maskDataset(dataset, inTrainSet),
    test: await maskDataset(dataset, inTestSet),
    split_idxs: inTrainSet,
  };
}

const g = addInverseRelation(
  graphFromPredicates("0 < 1", "1 < 2", "2 < 3", "2 < 4"),
  "<",
  ">",
);

trainTestSplit(graphToTFJSData(g), 0.7).then((dataset) => {
  const model = makeMLPPredicateModel(
    g.relationTypes().size,
    g.nodes.size,
    1024,
    1, // layers
    0.0, // l2
    0.0, // dropout
    1e-4, //learning rate
  );
  trainModel(model, dataset);
});
