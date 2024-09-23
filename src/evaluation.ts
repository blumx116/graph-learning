import * as tf from "@tensorflow/tfjs";
import { DataSet, TrainTestDataset } from "./model.ts";
import { Graph, Edge } from "./graph";

function maskGraph(graph: Graph, predicate: boolean[]): Graph {
  const edgesToKeep: Edge[] = Array.from(graph.edges).filter(
    (_, index) => predicate[index],
  );
  const result: Graph = new Graph();

  edgesToKeep.forEach((edge) =>
    result.addEdgeUnsafe(edge.fromNode, edge.relation, edge.toNode),
  );

  return result;
}

function EdgeWiseEvaluation(
  model: tf.LayersModel,
  dataset: DataSet,
  graph: Graph,
): any {
  const yhats: tf.Tensor = (model.call(dataset.xs, {}) as tf.Tensor[])[0];
  const result: Record<string, number> = {};

  yhats.print();
  alert("Go check your print statements, BOY");

  predictedProbs: tf.Tensor = tf.mul(yhats, dataset.ys);

  for (const edgeType: string in graph.relationTypes()) {
  }
}

function SplitEvaluation(
  model: tf.LayersModel,
  dataset: TrainTestDataset,
  graph: Graph,
): any {
  return {
    train: EdgeWiseEvaluation(
      model,
      dataset.train,
      maskGraph(graph, dataset.split_idxs),
    ),
    test: EdgeWiseEvaluation(
      model,
      dataset.test,
      maskGraph(
        graph,
        dataset.split_idxs.map((val) => !val),
      ),
    ),
  };
}
