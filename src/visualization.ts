import { Edge, Graph } from "./graph";
import { default as cytoscape } from "cytoscape";

const CURVE_STYLES: string[] = ["bezier", "segments", "round-segments"];
const LAYOUT = {
  name: "cose",
  componentSpacing: 40,
};

interface CytoScapeNode {
  data: {
    id: string;
  };
}

interface CytoScapeEdge {
  data: {
    id: string;
    source: string;
    target: string;
    relation: string;
    prob?: number;
    probColor?: string;
    isTrain?: boolean;
  };
}

function cytoScapeNode(node: string): CytoScapeNode {
  return { data: { id: node } };
}

function cytoScapeEdge(edge: Edge, idx: number): CytoScapeEdge {
  return {
    data: {
      id: edge.fromNode + edge.relation + edge.toNode + "--" + idx.toString(),
      source: edge.fromNode,
      target: edge.toNode,
      relation: edge.relation,
    },
  };
}

export function graphToCytoScapeElements(
  graph: Graph,
): (CytoScapeNode | CytoScapeEdge)[] {
  // TODO: output type is actually fairly well structured, I just don't want to define it
  const nodes: CytoScapeNode[] = Array.from(graph.nodes).map(cytoScapeNode);
  const edges: CytoScapeEdge[] = Array.from(graph.edges).map((edge, idx) =>
    cytoScapeEdge(edge, idx),
  );
  return [...nodes, ...edges];
}

function relationToCurveStyle(graph: Graph): Map<string, string> {
  const relations: string[] = Array.from(graph.relationTypes());
  if (relations.length > CURVE_STYLES.length) {
    alert(
      `Can't handle more than ${CURVE_STYLES.length} types of relations right now b/c I haven't figured out how to visualize it`,
    );
    throw Error("Too many relation types");
  }

  return new Map(
    relations.sort().map((relation, index) => [relation, CURVE_STYLES[index]]),
  );
}

function graphCurveStyles(graph: Graph): any[] {
  return Array.from(relationToCurveStyle(graph).entries()).map(
    ([relation, curveStyle]) => ({
      selector: `edge[relation="${relation}"]`,
      style: {
        "curve-style": curveStyle,
        ...(curveStyle !== "bezier" && {
          "segment-distances": [-5, 5, -5, 5, -5],
          "segment-weights": [0.16, 0.32, 0.48, 0.64, 0.8],
        }),
      },
    }),
  );
}

export function graphToCytoScapeStyle(graph: Graph): any[] {
  return [
    {
      selector: "node",
      style: {
        "background-color": "#777",
        label: "data(id)",
      },
    },
    {
      selector: "edge",
      style: {
        width: 3,
        "line-color": "#ccc",
        "target-arrow-color": "#ccc",
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
      },
    },
    {
      selector: "edge[?isTrain]",
      style: {
        "line-style": "dashed",
      },
    },
    {
      selector: "edge[prob>0]",
      style: {
        "line-color": "data(probColor)",
        "target-arrow-color": "data(probColor)",
      },
    },
    ...graphCurveStyles(graph),
  ];
}

function drawGraph(graph: Graph, container: HTMLElement): cytoscape.Core {
  return cytoscape({
    container: container,
    elements: graphToCytoScapeElements(graph),
    style: graphToCytoScapeStyle(graph),
    layout: LAYOUT,
  });
}