from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog, messagebox
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .solver import solve


@dataclass
class Node:
    idx: int
    x: float
    y: float
    bias: float = 0.0
    item: Optional[int] = None
    label: Optional[int] = None


@dataclass
class Edge:
    i: int
    j: int
    weight: float
    item: Optional[int] = None


class GraphEditor:
    def __init__(self, title: str = "qanneal Graph Editor"):
        self.root = tk.Tk()
        self.root.title(title)

        self.canvas = tk.Canvas(self.root, width=900, height=600, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(self.root)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        self.info = tk.Text(self.sidebar, width=40, height=20)
        self.info.pack(padx=8, pady=8)
        self.info.insert(tk.END, "Left-click empty: add node\n")
        self.info.insert(tk.END, "Left-click node: select/unselect\n")
        self.info.insert(tk.END, "Key 'b': set bias on selected node\n")
        self.info.insert(tk.END, "Key 'w': set weight between 2 selected nodes\n")
        self.info.insert(tk.END, "Key 'd' or Delete: delete selected\n")
        self.info.insert(tk.END, "Buttons: export/run/clear\n")

        btn_frame = tk.Frame(self.sidebar)
        btn_frame.pack(padx=8, pady=8, fill=tk.X)

        tk.Button(btn_frame, text="Export QUBO", command=self.export_qubo).pack(fill=tk.X)
        tk.Button(btn_frame, text="Run Solve (SQA)", command=self.run_solve).pack(fill=tk.X, pady=4)
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(fill=tk.X)

        self.nodes: List[Node] = []
        self.edges: Dict[Tuple[int, int], Edge] = {}
        self.selected: List[int] = []
        self.radius = 18

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("b", self.set_bias)
        self.root.bind("w", self.set_weight)
        self.root.bind("d", self.delete_selected)
        self.root.bind("<Delete>", self.delete_selected)

    def log(self, msg: str) -> None:
        self.info.insert(tk.END, msg + "\n")
        self.info.see(tk.END)

    def on_click(self, event) -> None:
        node_idx = self.find_node_at(event.x, event.y)
        if node_idx is None:
            self.add_node(event.x, event.y)
        else:
            self.toggle_select(node_idx)

    def find_node_at(self, x: float, y: float) -> Optional[int]:
        for node in self.nodes:
            dx = node.x - x
            dy = node.y - y
            if (dx * dx + dy * dy) <= self.radius * self.radius:
                return node.idx
        return None

    def add_node(self, x: float, y: float) -> None:
        idx = len(self.nodes)
        node = Node(idx=idx, x=x, y=y)
        node.item = self.canvas.create_oval(
            x - self.radius, y - self.radius,
            x + self.radius, y + self.radius,
            outline="black", width=2, fill="#f5f5f5"
        )
        node.label = self.canvas.create_text(x, y, text=str(idx))
        self.nodes.append(node)
        self.log(f"Added node {idx}")

    def toggle_select(self, idx: int) -> None:
        if idx in self.selected:
            self.selected.remove(idx)
        else:
            self.selected.append(idx)
        self.refresh_selection()

    def refresh_selection(self) -> None:
        for node in self.nodes:
            if node.item is None:
                continue
            color = "red" if node.idx in self.selected else "black"
            self.canvas.itemconfig(node.item, outline=color)

    def set_bias(self, _event=None) -> None:
        if len(self.selected) != 1:
            messagebox.showinfo("Bias", "Select exactly one node.")
            return
        idx = self.selected[0]
        node = self.nodes[idx]
        value = simpledialog.askfloat("Bias", f"Bias for node {idx}:", initialvalue=node.bias)
        if value is None:
            return
        node.bias = float(value)
        self.log(f"Set bias: node {idx} = {node.bias}")

    def set_weight(self, _event=None) -> None:
        if len(self.selected) != 2:
            messagebox.showinfo("Weight", "Select exactly two nodes.")
            return
        i, j = sorted(self.selected)
        key = (i, j)
        current = self.edges[key].weight if key in self.edges else 0.0
        value = simpledialog.askfloat("Weight", f"Weight for edge ({i},{j}):", initialvalue=current)
        if value is None:
            return
        self.add_edge(i, j, float(value))

    def add_edge(self, i: int, j: int, weight: float) -> None:
        key = (i, j)
        ni, nj = self.nodes[i], self.nodes[j]
        if key in self.edges:
            edge = self.edges[key]
            edge.weight = weight
            self.log(f"Updated edge ({i},{j}) weight = {weight}")
            return

        item = self.canvas.create_line(ni.x, ni.y, nj.x, nj.y, fill="#888", width=2)
        edge = Edge(i=i, j=j, weight=weight, item=item)
        self.edges[key] = edge
        self.canvas.tag_lower(item)
        self.log(f"Added edge ({i},{j}) weight = {weight}")

    def delete_selected(self, _event=None) -> None:
        if not self.selected:
            return
        remove = set(self.selected)
        self.selected = []

        # Remove edges
        for key in list(self.edges.keys()):
            if key[0] in remove or key[1] in remove:
                edge = self.edges.pop(key)
                if edge.item is not None:
                    self.canvas.delete(edge.item)

        # Remove nodes
        survivors: List[Node] = []
        old_to_new: Dict[int, int] = {}
        for node in self.nodes:
            if node.idx in remove:
                if node.item is not None:
                    self.canvas.delete(node.item)
                if node.label is not None:
                    self.canvas.delete(node.label)
            else:
                old_to_new[node.idx] = len(survivors)
                survivors.append(node)

        # Reindex nodes
        for node in survivors:
            node.idx = old_to_new[node.idx]
            if node.label is not None:
                self.canvas.itemconfig(node.label, text=str(node.idx))
        self.nodes = survivors

        # Rebuild edges with new indices
        new_edges: Dict[Tuple[int, int], Edge] = {}
        for edge in self.edges.values():
            if edge.i in remove or edge.j in remove:
                continue
            i = old_to_new[edge.i]
            j = old_to_new[edge.j]
            key = tuple(sorted((i, j)))
            new_edges[key] = Edge(i=key[0], j=key[1], weight=edge.weight, item=edge.item)
        self.edges = new_edges

        self.refresh_selection()
        self.log("Deleted selected nodes.")

    def export_qubo(self) -> np.ndarray:
        n = len(self.nodes)
        Q = np.zeros((n, n), dtype=float)
        for node in self.nodes:
            Q[node.idx, node.idx] += node.bias
        for (i, j), edge in self.edges.items():
            Q[i, j] += edge.weight
            Q[j, i] += edge.weight
        self.log("Exported QUBO matrix.")
        self.log(str(Q))
        return Q

    def run_solve(self) -> None:
        if len(self.nodes) == 0:
            messagebox.showinfo("Solve", "Add at least one node.")
            return
        Q = self.export_qubo()
        result = solve(Q, method="sqa", reads=5, return_bits=True)
        self.log(f"Solve best energy: {result.best_energy}")
        self.log(f"Best sample: {result.best_sample}")

    def clear(self) -> None:
        self.canvas.delete("all")
        self.nodes.clear()
        self.edges.clear()
        self.selected.clear()
        self.log("Cleared graph.")

    def run(self) -> None:
        self.root.mainloop()


def launch_graph_editor() -> None:
    GraphEditor().run()
