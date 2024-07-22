import assert from "assert";
import { Queue } from "typescript-collections";

export class Edge {
  fromNode: string;
  relation: string;
  toNode: string;

  public constructor(fromNode: string, relation: string, toNode: string) {
    this.fromNode = fromNode;
    this.relation = relation;
    this.toNode = toNode;
  }

  public toString(): string {
    return `${this.fromNode} ${this.relation} ${this.toNode}`;
  }
}

export class Graph {
  nodes: Set<string>;
  edges: Set<Edge>;
  // TODO: conceptually, these are sets
  // in practice, I'm constantly converting them to Arrays

  public constructor(nodes: Iterable<string> = [], edges: Iterable<Edge> = []) {
    this.nodes = new Set(nodes);
    this.edges = new Set(edges);

    for (const edge of this.edges) {
      assert(this.nodes.has(edge.fromNode));
      assert(this.nodes.has(edge.toNode));
    }
  }

  public addEdgeUnsafe(
    fromNode: string,
    relation: string,
    toNode: string,
  ): void {
    if (!this.nodes.has(fromNode)) {
      this.nodes.add(fromNode);
    }
    if (!this.nodes.has(toNode)) {
      this.nodes.add(toNode);
    }
    this.edges.add(new Edge(fromNode, relation, toNode));
  }

  public joinWithRename(
    other: Graph,
    naming_fn: (idx: number) => string = idx_naming,
  ): Graph {
    const nodeRename: Map<string, string> = new Map();

    for (const node of other.nodes) {
      const newName: string = naming_fn(this.nodes.size);
      nodeRename.set(node, newName);
      assert(!this.nodes.has(newName));
      this.nodes.add(newName);
    }

    for (const edge of other.edges) {
      this.edges.add(
        new Edge(
          nodeRename.get(edge.fromNode)!,
          edge.relation,
          nodeRename.get(edge.toNode)!,
        ),
      );
    }

    return this;
  }

  public relationTypes(): Set<string> {
    return new Set(Array.from(this.edges).map((edge) => edge.relation));
  }

  public toString(): string {
    return "Graph:\n" + this._printNodes() + "\n" + this._printEdges();
  }

  private _printNodes(): string {
    return " nodes:\n  " + Array.from(this.nodes).join("\n  ");
  }

  private _printEdges(): string {
    return (
      " edges:\n  " +
      Array.from(this.edges)
        .map((edge) => edge.toString())
        .join("\n  ")
    );
  }
}

/*
 * Creates a new graph object from a human readable set of edges, described in infix notation
 * NOTE: there is no way to create a node with no edges using this function
 * NOTE: multigraphs are allowed and no sanity checks are performed
 *
 * @param {edges}: each edge should be a string like "fromNode relationType toNode"
 * 	i.e. "Carter isChildOf Debbie"
 * 	any non-space character is valid for each of the names in the string
 *
 * @return {graph}: the newly created graph
 */
export function graphFromPredicates(...edges: string[]): Graph {
  const graph: Graph = new Graph();

  for (const edgeString of edges) {
    const [fromNode, relationType, toNode] = edgeString.split(" ");
    graph.addEdgeUnsafe(fromNode, relationType, toNode);
  }

  return graph;
}

/*
 * Adds new edges to the specified graph, denoting (non-unique) inverses of an original edge type
 * For instance, given a graph with edges all denoting the 'less-than' relation
 * addInverseRelation(g, 'less-than', 'greater-than')
 * would add new nodes of type 'greater-than' pointing in the opposite direction
 *
 * @param {graph}: the graph to be modified *in place*
 * @param {originalRelation}: new inverse relations will be added for each relation
 * 	of type 'originalRelation' in the graph
 * @param {reversedRelation}: relationType for newly created edges
 * 	for simplicity, it is assumed that there are no edges of this relationType in the graph when the
 * 	function is initially called. This restriction is primarily in place to make the calling semantics more obvious
 * 	but could be relaxed in the future if the function were made symmetric to both arguments
 */
export function addInverseRelation(
  graph: Graph,
  originalRelation: string,
  reversedRelation: string,
): Graph {
  assert(!graph.relationTypes().has(reversedRelation));

  const edgesToInvert: Edge[] = Array.from(graph.edges).filter(
    (edge) => edge.relation == originalRelation,
  );

  for (const edge of edgesToInvert) {
    graph.edges.add(new Edge(edge.toNode, reversedRelation, edge.fromNode));
  }

  return graph;
}

export function idx_naming(idx: number): string {
  return idx.toString();
}

function _normalizeDistribution<T>(
  distribution: Map<number, T>,
): Map<number, T> {
  Array.from(distribution.keys()).forEach((prob) => assert(prob >= 0));

  const total: number = Array.from(distribution.keys()).reduce(
    (a, b) => a + b,
    0,
  );

  return new Map(
    Array.from(distribution.entries()).map(([prob, value]) => [
      prob / total,
      value,
    ]),
  );
}

function _expectedValue(distribution: Map<number, number>) {
  distribution = _normalizeDistribution(distribution);
  return Array.from(distribution.entries()).reduce(
    (total, [prob, value]) => total + prob * value,
    0,
  );
}

/*
 * Randomly samples from a distribution in linear time
 * Could use something smarter (like the Alias method), but that seems like premature optimization
 *
 * @param {distribution}: probability => value
 * 	probability distribution doesn't need to be normalized
 *
 * @returns {value}: randomly selected value
 */
export function sampleFromDistribution<T>(distribution: Map<number, T>) {
  distribution = _normalizeDistribution(distribution);

  const random_value: number = Math.random();
  var cumulative_prob: number = 0;

  for (const [prob, value] of Array.from(distribution)) {
    cumulative_prob += prob;
    if (cumulative_prob >= random_value) {
      return value;
    }
  }
  throw Error("Unreachable code path");
}

/*
 * Recursively generates a tree
 * The number of children for each node is randomly sampled from child_probs
 *
 * @param {child_probs}: the probability distribution for the number of children at each layer
 * 	NOTE: if the expected number of children is >1, then max depth must be set to prevent arbitrary recursion
 *  of the form probability => value
 * @param {max_depth}: if provided, the tree will have depth no greater than max depth
 * @param {naming_fn}: is called with argument idx = i to name the `ith` node, starting from 0
 *
 * @returns {graph}:
 */
export function randomTreeGraph(
  child_probs: Map<number, number>,
  relation: string,
  max_depth?: number,
  naming_fn: (idx: number) => string = idx_naming,
) {
  assert(_expectedValue(child_probs) < 1 || max_depth !== undefined);
  assert(max_depth === undefined || max_depth >= 0);

  const graph = new Graph();
  const q = new Queue<[string, number]>();

  var cur_node: string = naming_fn(graph.nodes.size);
  graph.nodes.add(cur_node);
  q.enqueue([cur_node, 1]);

  while (!q.isEmpty()) {
    const [node, depth] = q.dequeue()!;

    if (max_depth !== undefined && depth === max_depth) {
      // node can't have children because it is already at the max depth
      continue;
    }

    const n_children: number = sampleFromDistribution(child_probs);

    for (var i = 0; i < n_children; i++) {
      const child_name: string = naming_fn(graph.nodes.size);
      graph.nodes.add(child_name);
      q.enqueue([child_name, depth + 1]);
      graph.edges.add(new Edge(node, relation, child_name));
    }
  }

  return graph;
}

/*
 * Creates a graph composed of a number of random tree graphs
 *
 * @param {tree_count_probs}: probability distribution over the number of disconnected trees that the graph is composed of
 * @param {child_probs}: the probability distribution for the number of children at each layer
 * 	NOTE: if the expected number of children is >1, then max depth must be set to prevent arbitrary recursion
 *  of the form probability => value
 * @param {max_depth}: if provided, the tree will have depth no greater than max depth
 * @param {naming_fn}: is called with argument idx = i to name the `ith` node, starting from 0
 *
 * @returns {graph}: graph composed of randomly generated trees
 */
export function randomForestGraph(
  tree_count_probs: Map<number, number>,
  child_probs: Map<number, number>,
  relation: string,
  max_depth?: number,
  naming_fn: (idx: number) => string = idx_naming,
) {
  const base_graph: Graph = new Graph();
  const tree_count: number = sampleFromDistribution(tree_count_probs);
  for (var i = 0; i < tree_count; i++) {
    base_graph.joinWithRename(
      randomTreeGraph(child_probs, relation, max_depth, naming_fn),
    );
  }
  return base_graph;
}
