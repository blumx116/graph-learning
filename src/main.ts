import "./style.css";
import { Graph, graphFromPredicates } from "./graph.ts";
import { default as cytoscape } from "cytoscape";
import {
  graphToCytoScapeGraph,
  graphToCytoScapeElements,
  graphToCytoScapeStyle,
} from "./visualization.ts";

console.log("we're running!");

const PROB_COLOR_MAP: [number, string][] = [
  [0.2, "red"],
  [0.4, "orange"],
  [0.6, "yellow"],
  [0.8, "yellowgreen"],
  [1.0, "green"],
];

const options = {
  container: document.getElementById("vis-container"),
  elements: [
    // Nodes
    { data: { id: "a" } },
    { data: { id: "b" } },
    { data: { id: "c" } },
    { data: { id: "d" } },
    { data: { id: "e" } },
    { data: { id: "f" } },

    // Edges
    { data: { id: "ab", source: "a", target: "b", type: "solid" } },
    { data: { id: "bc", source: "b", target: "c", type: "dashed" } },
    { data: { id: "ca", source: "c", target: "a", type: "dotted" } },
    { data: { id: "ab2", source: "a", target: "b", type: "solid" } }, // Multiple edge
    { data: { id: "de", source: "d", target: "e", type: "solid" } },
    { data: { id: "ef", source: "e", target: "f", type: "dashed" } },
  ],
  style: [
    {
      selector: "node",
      style: {
        "background-color": "#666",
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
      selector: 'edge[type="dashed"]',
      style: {
        "line-style": "dashed",
      },
    },
    {
      selector: 'edge[type="dotted"]',
      style: {
        "line-style": "dotted",
      },
    },
  ],
  ,
};
const cy = cytoscape(options);

function displayVis() {
  const graphText: string = document
    .querySelector<HTMLTextAreaElement>("#freeform-graph")!
    .value.trim();
  const lines = graphText.split("\n").map((text) => text.trim());
  console.log(lines);
  const graph = graphFromPredicates(...lines);
  console.log(graphToCytoScapeGraph(graph));
  cy.elements().remove();
  cy.add(graphToCytoScapeElements(graph));
  addTrainTest(Array.from(graph.edges).map((_) => Math.random() < 0.7));
  addProbabilities(Array.from(graph.edges).map((_, i) => (i + 1) * 0.15));
  cy.style(graphToCytoScapeStyle(graph));
  cy.ready(function () {
    var layout = cy.layout({
      name: "cose",
      componentSpacing: 40,
    });
    layout.run();
  });
  console.log("new json");
}

function addTrainTest(trainTestSplit: boolean[]) {
  const n: number = cy;
  cy.elements("edge").forEach((elem, i) => {
    elem.data({ ...elem.data(), isTrain: trainTestSplit[i] });
  });
}

function probToColor(prob: number): string {
  for (const [threshold, color] of PROB_COLOR_MAP) {
    if (prob < threshold) {
      return color;
    }
  }
  throw Error("TODO: fill in this error message");
}

function addProbabilities(probs: number[]) {
  cy.elements("edge").forEach((elem, i) => {
    elem.data({
      ...elem.data(),
      prob: probs[i],
      probColor: probToColor(probs[i]),
    });
    console.log(elem.data());
  });
}

document.addEventListener("DOMContentLoaded", function () {
  document
    .querySelector<HTMLButtonElement>("#submit")!
    .addEventListener("click", displayVis);
});

console.log("Created vis?");

const g = new Graph();
console.log(g.toString());
