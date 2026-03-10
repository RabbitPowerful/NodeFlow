import inspect
import random
from abc import ABC, abstractmethod
import dearpygui.dearpygui as dpg
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import threading

def hex_to_rgb(hex_color: str, alpha: int = 255) -> tuple:
    """Convert hex color string to (R, G, B, A) tuple.
    Accepts '#RRGGBB' or 'RRGGBB' format.
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r, g, b, alpha)

class NodeEditorTheme:
    """Centralises all DPG theme calls. Use _s / _c helpers so missing
    constants in older DPG versions are silently ignored."""

    @staticmethod
    def _c(name: str, value: tuple):
        if hasattr(dpg, name):
            dpg.add_theme_color(getattr(dpg, name), value,
                                category=dpg.mvThemeCat_Nodes)

    @staticmethod
    def _s(name: str, value):
        if hasattr(dpg, name):
            dpg.add_theme_style(getattr(dpg, name), value,
                                category=dpg.mvThemeCat_Nodes)

    @classmethod
    def apply_global(cls):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvNodeEditor):
                cls._c("mvNodeCol_GridBackground",     (237, 237, 240, 255))  # warm light gray
                cls._c("mvNodeCol_GridLine",           (237, 237, 240, 255))  # same as bg = invisible grid
                cls._c("mvNodeCol_Link",               ( 40,  40,  45, 200))  # near-black thin wires
                cls._c("mvNodeCol_LinkHovered",        ( 80, 180, 170, 255))  # teal on hover (matches ref)
                cls._c("mvNodeCol_LinkSelected",       ( 80, 180, 170, 255))  # teal when selected
                cls._c("mvNodeCol_BoxSelector",        ( 80, 180, 170,  30))
                cls._c("mvNodeCol_BoxSelectorOutline", ( 80, 180, 170, 160))
                cls._s("mvNodesStyleVar_GridSpacing",               28)
                cls._s("mvNodesStyleVar_LinkThickness",              1)        # thin like reference
                cls._s("mvNodesStyleVar_LinkLineSegmentsPerLength",  0.03)    # smooth bezier
                cls._s("mvNodesStyleVar_LinkHoverDistance",         12)
                cls._s("mvNodesStyleVar_PinCircleRadius",            4)
                cls._s("mvNodesStyleVar_PinLineThickness",           1)
                cls._s("mvNodesStyleVar_PinHoverRadius",            10)
                cls._s("mvNodesStyleVar_PinOffset",                  0)
                
            with dpg.theme_component(dpg.mvNode):
                cls._c("mvNodeCol_TitleBar",                (40,  90, 140, 255))
                cls._c("mvNodeCol_TitleBarHovered",         (55, 115, 175, 255))
                cls._c("mvNodeCol_TitleBarSelected",        (70, 150, 220, 255))
                cls._c("mvNodeCol_NodeBackground",          (30,  30,  45, 230))
                cls._c("mvNodeCol_NodeBackgroundHovered",   (40,  40,  60, 240))
                cls._c("mvNodeCol_NodeBackgroundSelected",  (50,  50,  75, 255))
                cls._c("mvNodeCol_NodeOutline",             (80,  80, 110, 180))
                cls._c("mvNodeCol_Pin",                     (100, 200, 255, 220))
                cls._c("mvNodeCol_PinHovered",              (180, 240, 255, 255))
                cls._s("mvNodesStyleVar_NodeCornerRounding",    6)
                cls._s("mvNodesStyleVar_NodePaddingHorizontal", 12)
                cls._s("mvNodesStyleVar_NodePaddingVertical",    8)
                cls._s("mvNodesStyleVar_NodeBorderThickness",    1)

        dpg.bind_theme(theme)

    @classmethod
    def apply_to_node(cls, node_id: int, title_color: tuple):
        brighter  = tuple(min(v + 25, 255) for v in title_color)
        brightest = tuple(min(v + 50, 255) for v in title_color)
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvNode):
                cls._c("mvNodeCol_TitleBar",         title_color)
                cls._c("mvNodeCol_TitleBarHovered",  brighter)
                cls._c("mvNodeCol_TitleBarSelected", brightest)
        dpg.bind_item_theme(node_id, t)

    @staticmethod
    def print_available_constants():
        for name in sorted(dir(dpg)):
            if name.startswith(("mvNodeCol_", "mvNodesStyleVar_")):
                print(f"  dpg.{name} = {getattr(dpg, name)}")

# Base Node
class BaseNode(ABC):
    """
    Subclass this to create any new node type.

    Minimal subclass:
        class MyNode(BaseNode):
            LABEL       = "My Node"
            TITLE_COLOR = (R, G, B, 255)

            def execute(self, **kwargs):
                ...
                return result
    """

    LABEL: str       = "Node"
    TITLE_COLOR: tuple = (40, 90, 140, 255)

    def __init__(self):
        self.node_id:      int | None = None
        self.input_attrs:  dict[str, int] = {}
        self.output_attr:  int | None = None          # kept for single-output nodes
        self.output_attrs: dict[str, int] = {} 

    # ── Introspection ──────────────────────────────────────────────────────
    def _get_params(self) -> list[str]:
        """Return the parameter names for this node's execute() method,
        excluding 'self'."""
        sig = inspect.signature(self.execute)
        return [p for p in sig.parameters if p != "self"]

    # ── DPG build ──────────────────────────────────────────────────────────
    def build(self, parent: str | int, pos: tuple = (10, 10)) -> int:
        """Create the DPG node widget and register all pins. Returns node_id."""
        params = self._get_params()

        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:
            for name in params:
                with dpg.node_attribute(
                    attribute_type=dpg.mvNode_Attr_Input
                ) as attr_id:
                    dpg.add_text(name)
                self.input_attrs[name] = attr_id

            with dpg.node_attribute(
                attribute_type=dpg.mvNode_Attr_Output
            ) as self.output_attr:
                dpg.add_text("output")

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    # ── Execution ──────────────────────────────────────────────────────────
    @abstractmethod
    def execute(self, **kwargs):
        """Override this with the node's logic. Keyword args match input pins."""
        ...

# My Custom Nodes
class TerminalNode(BaseNode):
    LABEL       = "Terminal"
    TITLE_COLOR = (20, 20, 20, 255)

    MAX_LINES = 200
    WIDTH     = 340

    def __init__(self):
        super().__init__()
        self._lines   = []
        self._list_id = None

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as attr_id:
                dpg.add_text("val")
            self.input_attrs["val"] = attr_id

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                self._list_id = dpg.add_listbox(items=[], width=self.WIDTH, num_items=8)
                dpg.add_button(label="Clear", width=self.WIDTH, callback=self._clear)

            self.output_attr = None  # sink node — no output pin

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        self._apply_theme()
        return self.node_id

    def _apply_theme(self):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvListbox):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,      (10, 12, 10, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text,          (0, 255, 80, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,   (10, 12, 10, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (0, 180, 60, 180))
        dpg.bind_item_theme(self._list_id, t)

    def _clear(self):
        self._lines.clear()
        dpg.configure_item(self._list_id, items=[])

    def execute(self, val=None):
        self._lines.append(str(val))
        if len(self._lines) > self.MAX_LINES:
            self._lines.pop(0)
        dpg.configure_item(self._list_id, items=list(self._lines))
        dpg.set_value(self._list_id, str(val))  # auto-scroll to latest
        return val

class CSVNode(BaseNode):
    
    LABEL       = "CSV Loader"
    TITLE_COLOR = (30, 80, 120, 255)

    WIDTH = 280

    def __init__(self):
        super().__init__()
        self._df          = None
        self._path        = None
        self._info_id     = None   # text widget showing shape + columns
        self._preview_id  = None   # listbox showing first few rows

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Static body ──────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_button(
                    label="Browse CSV…",
                    width=self.WIDTH,
                    callback=self._open_dialog,
                )
                self._info_id = dpg.add_text(
                    "No file loaded",
                    color=(160, 160, 160, 255),
                )
                self._preview_id = dpg.add_listbox(
                    items=[],
                    width=self.WIDTH,
                    num_items=5,
                )

            # ── Output pin — emits a DataFrame ───────────────────────────
            with dpg.node_attribute(
                attribute_type=dpg.mvNode_Attr_Output
            ) as self.output_attr:
                dpg.add_text("data")

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        self._register_dialog()
        return self.node_id

    # ── File dialog ───────────────────────────────────────────────────────
    def _register_dialog(self):
        """Create a file dialog — registered once per node instance."""
        with dpg.file_dialog(
            label="Select CSV",
            width=500,
            height=350,
            show=False,
            callback=self._on_file_selected,
            tag=f"csv_dialog_{self.node_id}",   # unique tag per instance
        ):
            dpg.add_file_extension(".csv", color=(0, 255, 120, 255))
            dpg.add_file_extension(".*",   color=(200, 200, 200, 255))

    def _open_dialog(self):
        dpg.show_item(f"csv_dialog_{self.node_id}")

    def _on_file_selected(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if not path:
            return
        try:
            self._df   = pd.read_csv(path)
            self._path = path
            self._update_preview()
        except Exception as e:
            dpg.set_value(self._info_id, f"Error: {e}")

    # ── Preview update ────────────────────────────────────────────────────
    def _update_preview(self):
        if self._df is None:
            return

        filename = self._path.split("\\")[-1].split("/")[-1]
        r, c     = self._df.shape
        dpg.set_value(
            self._info_id,
            f"{filename}  |  {r} rows  {c} cols",
        )

        # Header row + first 4 data rows
        header  = " | ".join(self._df.columns.astype(str))
        rows    = [
            " | ".join(str(v) for v in row)
            for row in self._df.head(4).values
        ]
        dpg.configure_item(self._preview_id, items=[header, *rows])

    # ── Execution — just passes the DataFrame downstream ─────────────────
    def execute(self):
        return self._df

class ColumnSelectorNode(BaseNode):
    LABEL       = "Column Selector"
    TITLE_COLOR = (20, 100, 110, 255)

    WIDTH = 260

    def __init__(self):
        super().__init__()
        self._graph         = None
        self._all_columns:  list[str] = []
        self._x_columns:    list[str] = []
        self._all_list_id:  int | None = None
        self._x_list_id:    int | None = None
        self._y_combo_id:   int | None = None
        self._status_id:    int | None = None

    def set_graph(self, graph):
        self._graph = graph

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Input pin ─────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as attr_id:
                dpg.add_text("data")
            self.input_attrs["data"] = attr_id

            # ── Static body ───────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                self._status_id = dpg.add_text(
                    "Connect a CSV node then hit Refresh",
                    color=(160, 160, 160, 255),
                )
                dpg.add_button(
                    label="↺ Refresh Columns",
                    width=self.WIDTH,
                    callback=self._refresh_from_upstream,
                )
                dpg.add_spacer(height=4)

                # Available columns
                dpg.add_text("Available columns:", color=(180, 180, 180, 255))
                self._all_list_id = dpg.add_listbox(
                    items=[],
                    width=self.WIDTH,
                    num_items=5,
                )
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="→ Add to X",
                        width=self.WIDTH // 2 - 2,
                        callback=self._add_to_x,
                    )
                    dpg.add_button(
                        label="✕ Remove X",
                        width=self.WIDTH // 2 - 2,
                        callback=self._remove_from_x,
                    )

                # Selected X features
                dpg.add_text("X features:", color=(100, 200, 255, 255))
                self._x_list_id = dpg.add_listbox(
                    items=[],
                    width=self.WIDTH,
                    num_items=4,
                )

                # y target
                dpg.add_text("y target:", color=(255, 180, 80, 255))
                self._y_combo_id = dpg.add_combo(
                    items=[],
                    width=self.WIDTH,
                )

            # ── Output pins ───────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as x_attr:
                dpg.add_text("X")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as y_attr:
                dpg.add_text("y")

            self.output_attrs = {"X": x_attr, "y": y_attr}
            self.output_attr  = None

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    # ── Refresh from upstream ─────────────────────────────────────────────
    def _refresh_from_upstream(self):
        """Pull the DataFrame from the connected upstream node directly,
        without needing to run the full graph."""
        if self._graph is None:
            dpg.set_value(self._status_id, "No graph reference.")
            return

        # Find which output attr is linked to our data input pin
        in_attr = self.input_attrs.get("data")
        source_out = None
        for _, (out_a, in_a) in self._graph._links.items():
            if in_a == in_attr:
                source_out = out_a
                break

        if source_out is None:
            dpg.set_value(self._status_id, "No upstream node connected.")
            return

        # Find the upstream node and call execute() on it directly
        source_nid = self._graph._attr_to_node.get(source_out)
        source_node = self._graph._nodes.get(source_nid)

        if source_node is None:
            dpg.set_value(self._status_id, "Upstream node not found.")
            return

        try:
            df = source_node.execute()
        except Exception as e:
            dpg.set_value(self._status_id, f"Error: {e}")
            return

        if df is None:
            dpg.set_value(self._status_id, "Upstream returned None — load a CSV first.")
            return

        self._load_columns(list(df.columns))

    def _load_columns(self, columns: list[str]):
        """Populate widgets with column names, preserving existing selections."""
        prev_x = list(self._x_columns)
        prev_y = dpg.get_value(self._y_combo_id)

        self._all_columns = columns

        # Restore valid previous selections, drop any that no longer exist
        self._x_columns = [c for c in prev_x if c in columns]

        dpg.configure_item(self._all_list_id, items=self._all_columns)
        dpg.configure_item(self._x_list_id,   items=self._x_columns)
        dpg.configure_item(self._y_combo_id,  items=self._all_columns)

        if prev_y in columns:
            dpg.set_value(self._y_combo_id, prev_y)

        dpg.set_value(self._status_id, f"{len(columns)} columns loaded")

    # ── Column management ─────────────────────────────────────────────────
    def _add_to_x(self):
        selected = dpg.get_value(self._all_list_id)
        if selected and selected not in self._x_columns:
            self._x_columns.append(selected)
            dpg.configure_item(self._x_list_id, items=list(self._x_columns))

    def _remove_from_x(self):
        selected = dpg.get_value(self._x_list_id)
        if selected in self._x_columns:
            self._x_columns.remove(selected)
            dpg.configure_item(self._x_list_id, items=list(self._x_columns))

    # ── Execution ─────────────────────────────────────────────────────────
    def execute(self, data=None):
        if data is None:
            return {"X": None, "y": None}

        # Refresh columns if data changed
        if list(data.columns) != self._all_columns:
            self._load_columns(list(data.columns))

        y_col = dpg.get_value(self._y_combo_id)
        X = data[self._x_columns] if self._x_columns else None
        y = data[y_col]           if y_col in data.columns else None

        return {"X": X, "y": y}


class ANNNode(BaseNode):
    LABEL       = "ANN"
    TITLE_COLOR = (50, 90, 160, 255)
    WIDTH       = 340

    def __init__(self):
        super().__init__()

        # ── Layer config ──────────────────────────────────────────────────
        self._layers        = []
        self._layer_list_id = None
        self._units_id      = None
        self._activation_id = None

        # ── Tab state ─────────────────────────────────────────────────────
        self._tab_btns      = {}
        self._tab_groups    = {}
        self._active_tab    = "Layers"

        # ── Training config ───────────────────────────────────────────────
        self._task_id       = None
        self._optimizer_id  = None
        self._lr_id         = None
        self._epochs_id     = None
        self._batch_id      = None
        self._val_split_id  = None

        # ── Regularization config ─────────────────────────────────────────
        self._dropout_id    = None
        self._l1_id         = None
        self._l2_id         = None
        self._wd_id         = None
        self._early_stop_id = None
        self._patience_id   = None

        # ── Status / progress ─────────────────────────────────────────────
        self._status_id     = None
        self._progress_id   = None

        # ── Runtime ───────────────────────────────────────────────────────
        self._model         = None
        self._scaler_X      = None
        self._scaler_y      = None
        self._is_training   = False
        self._result        = None
        self._last_X        = None
        self._last_y        = None

    # ══════════════════════════════════════════════════════════════════════
    #  BUILD
    # ══════════════════════════════════════════════════════════════════════
    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Input pins ────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("X")
            self.input_attrs["X"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("y")
            self.input_attrs["y"] = a

            # ── Body ──────────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)

                # Manual tab buttons
                with dpg.group(horizontal=True):
                    for tab_name in ["Layers", "Training", "Regularization"]:
                        btn = dpg.add_button(
                            label=tab_name,
                            width=self.WIDTH // 3,
                            callback=self._on_tab_click,
                            user_data=tab_name,
                        )
                        self._tab_btns[tab_name] = btn

                dpg.add_spacer(height=8)

                # Layers tab content
                with dpg.group(show=True) as g:
                    self._build_layers_tab()
                self._tab_groups["Layers"] = g

                # Training tab content
                with dpg.group(show=False) as g:
                    self._build_training_tab()
                self._tab_groups["Training"] = g

                # Regularization tab content
                with dpg.group(show=False) as g:
                    self._build_regularization_tab()
                self._tab_groups["Regularization"] = g

                dpg.add_spacer(height=8)

                # Status + progress
                self._status_id   = dpg.add_text(
                    "Ready", color=hex_to_rgb("#555555"))
                self._progress_id = dpg.add_progress_bar(
                    default_value=0.0, width=self.WIDTH)
                dpg.add_spacer(height=4)

                # Train button
                train_btn = dpg.add_button(
                    label="TRAIN",
                    width=self.WIDTH,
                    height=38,
                    callback=self._on_train_click,
                )
                self._apply_btn_theme(train_btn, hex_to_rgb("#2D6A9F"))

            # ── Output pins ───────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as a:
                dpg.add_text("predictions")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as b:
                dpg.add_text("metrics")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as c:
                dpg.add_text("model")

            self.output_attrs = {"predictions": a, "metrics": b, "model": c}
            self.output_attr  = None

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        self._refresh_tab_styles()
        return self.node_id

    # ══════════════════════════════════════════════════════════════════════
    #  TAB CONTENT — no dpg.tab() wrapper, just plain widgets
    # ══════════════════════════════════════════════════════════════════════
    def _build_layers_tab(self):
        dpg.add_text("Task:", color=hex_to_rgb("#333333"))
        self._task_id = dpg.add_combo(
            items=["Classification", "Regression"],
            default_value="Classification",
            width=self.WIDTH,
        )
        dpg.add_spacer(height=8)

        dpg.add_text("Hidden Layers:", color=hex_to_rgb("#333333"))
        self._layer_list_id = dpg.add_listbox(
            items=[],
            width=self.WIDTH,
            num_items=5,
        )
        dpg.add_spacer(height=6)

        with dpg.group(horizontal=True):
            dpg.add_text("Units:", color=hex_to_rgb("#555555"))
            self._units_id = dpg.add_input_int(
                default_value=64,
                min_value=1,
                max_value=4096,
                width=90,
            )

        dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            dpg.add_text("Act:  ", color=hex_to_rgb("#555555"))
            self._activation_id = dpg.add_combo(
                items=["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU", "GELU", "None"],
                default_value="ReLU",
                width=self.WIDTH - 52,
            )

        dpg.add_spacer(height=6)
        with dpg.group(horizontal=True):
            add_btn = dpg.add_button(
                label="+ Add Layer",
                width=self.WIDTH // 2 - 2,
                callback=self._add_layer,
            )
            rem_btn = dpg.add_button(
                label="- Remove Layer",
                width=self.WIDTH // 2 - 2,
                callback=self._remove_layer,
            )
            self._apply_btn_theme(add_btn, hex_to_rgb("#4A7C59"))
            self._apply_btn_theme(rem_btn, hex_to_rgb("#8B4444"))

    def _build_training_tab(self):
        dpg.add_text("Optimizer:", color=hex_to_rgb("#333333"))
        self._optimizer_id = dpg.add_combo(
            items=["Adam", "SGD", "RMSprop", "AdamW", "Adagrad"],
            default_value="Adam",
            width=self.WIDTH,
        )
        dpg.add_spacer(height=8)

        with dpg.group(horizontal=True):
            dpg.add_text("Learning Rate:", color=hex_to_rgb("#555555"))
            self._lr_id = dpg.add_input_float(
                default_value=0.001,
                format="%.5f",
                width=110,
            )
        with dpg.group(horizontal=True):
            dpg.add_text("Epochs:       ", color=hex_to_rgb("#555555"))
            self._epochs_id = dpg.add_input_int(
                default_value=100,
                min_value=1,
                width=110,
            )
        with dpg.group(horizontal=True):
            dpg.add_text("Batch Size:   ", color=hex_to_rgb("#555555"))
            self._batch_id = dpg.add_input_int(
                default_value=32,
                min_value=1,
                width=110,
            )
        with dpg.group(horizontal=True):
            dpg.add_text("Val Split:    ", color=hex_to_rgb("#555555"))
            self._val_split_id = dpg.add_input_float(
                default_value=0.2,
                format="%.2f",
                min_value=0.0,
                max_value=0.5,
                width=110,
            )

    def _build_regularization_tab(self):
        with dpg.group(horizontal=True):
            dpg.add_text("Dropout Rate: ", color=hex_to_rgb("#555555"))
            self._dropout_id = dpg.add_input_float(
                default_value=0.0,
                format="%.2f",
                min_value=0.0,
                max_value=0.9,
                width=110,
            )
        with dpg.group(horizontal=True):
            dpg.add_text("L1 Lambda:    ", color=hex_to_rgb("#555555"))
            self._l1_id = dpg.add_input_float(
                default_value=0.0,
                format="%.5f",
                width=110,
            )
        with dpg.group(horizontal=True):
            dpg.add_text("L2 Lambda:    ", color=hex_to_rgb("#555555"))
            self._l2_id = dpg.add_input_float(
                default_value=0.0,
                format="%.5f",
                width=110,
            )
        with dpg.group(horizontal=True):
            dpg.add_text("Weight Decay: ", color=hex_to_rgb("#555555"))
            self._wd_id = dpg.add_input_float(
                default_value=0.0,
                format="%.5f",
                width=110,
            )
        dpg.add_spacer(height=8)
        dpg.add_text("Early Stopping:", color=hex_to_rgb("#333333"))
        self._early_stop_id = dpg.add_checkbox(
            label="Enable", default_value=False)
        with dpg.group(horizontal=True):
            dpg.add_text("Patience:     ", color=hex_to_rgb("#555555"))
            self._patience_id = dpg.add_input_int(
                default_value=10,
                min_value=1,
                width=110,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  TAB SWITCHING
    # ══════════════════════════════════════════════════════════════════════
    def _on_tab_click(self, sender, app_data, user_data):
        self._active_tab = user_data
        for name, group in self._tab_groups.items():
            dpg.configure_item(group, show=(name == user_data))
        self._refresh_tab_styles()

    def _refresh_tab_styles(self):
        for name, btn in self._tab_btns.items():
            if name == self._active_tab:
                self._apply_btn_theme(btn, hex_to_rgb("#2D6A9F"))
            else:
                self._apply_btn_theme(btn, hex_to_rgb("#555555"))

    # ══════════════════════════════════════════════════════════════════════
    #  LAYER MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════
    def _layer_str(self, layer):
        return f"Linear({layer['units']})  {layer['activation']}"

    def _add_layer(self):
        units      = dpg.get_value(self._units_id)
        activation = dpg.get_value(self._activation_id)
        self._layers.append({"units": units, "activation": activation})
        dpg.configure_item(self._layer_list_id,
                           items=[self._layer_str(l) for l in self._layers])

    def _remove_layer(self):
        if not self._layers:
            return
        selected = dpg.get_value(self._layer_list_id)
        for i, layer in enumerate(self._layers):
            if self._layer_str(layer) == selected:
                self._layers.pop(i)
                break
        else:
            self._layers.pop()
        dpg.configure_item(self._layer_list_id,
                           items=[self._layer_str(l) for l in self._layers])

    # ══════════════════════════════════════════════════════════════════════
    #  MODEL BUILDING
    # ══════════════════════════════════════════════════════════════════════
    def _get_activation(self, name):
        return {
            "ReLU":      nn.ReLU(),
            "Sigmoid":   nn.Sigmoid(),
            "Tanh":      nn.Tanh(),
            "LeakyReLU": nn.LeakyReLU(),
            "ELU":       nn.ELU(),
            "GELU":      nn.GELU(),
            "None":      nn.Identity(),
        }.get(name, nn.ReLU())

    def _build_model(self, input_size, output_size):
        dropout = dpg.get_value(self._dropout_id)
        layers  = []
        prev    = input_size

        for layer in self._layers:
            layers.append(nn.Linear(prev, layer["units"]))
            layers.append(self._get_activation(layer["activation"]))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = layer["units"]

        layers.append(nn.Linear(prev, output_size))

        task = dpg.get_value(self._task_id)
        if task == "Classification" and output_size == 1:
            layers.append(nn.Sigmoid())
        elif task == "Classification" and output_size > 1:
            layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def _get_optimizer(self, model):
        name = dpg.get_value(self._optimizer_id)
        lr   = dpg.get_value(self._lr_id)
        wd   = dpg.get_value(self._wd_id)
        return {
            "Adam":    optim.Adam(model.parameters(),    lr=lr, weight_decay=wd),
            "SGD":     optim.SGD(model.parameters(),     lr=lr, weight_decay=wd),
            "RMSprop": optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd),
            "AdamW":   optim.AdamW(model.parameters(),   lr=lr, weight_decay=wd),
            "Adagrad": optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd),
        }.get(name, optim.Adam(model.parameters(), lr=lr))

    # ══════════════════════════════════════════════════════════════════════
    #  TRAINING
    # ══════════════════════════════════════════════════════════════════════
    def _set_status(self, text, color=None):
        color = color or hex_to_rgb("#555555")
        dpg.set_value(self._status_id, text)
        dpg.configure_item(self._status_id, color=color)

    def _on_train_click(self):
        if self._last_X is None or self._last_y is None:
            self._set_status("Connect X and y then run graph first.",
                             hex_to_rgb("#CC4444"))
            return
        if self._is_training:
            self._set_status("Already training...", hex_to_rgb("#CC8800"))
            return
        threading.Thread(
            target=self._train,
            args=(self._last_X, self._last_y),
            daemon=True,
        ).start()

    def _train(self, X_df, y_series):
        self._is_training = True
        self._set_status("Preparing data...", hex_to_rgb("#2266AA"))

        try:
            task        = dpg.get_value(self._task_id)
            epochs      = dpg.get_value(self._epochs_id)
            batch_size  = dpg.get_value(self._batch_id)
            val_split   = dpg.get_value(self._val_split_id)
            l1_lambda   = dpg.get_value(self._l1_id)
            l2_lambda   = dpg.get_value(self._l2_id)
            early_stop  = dpg.get_value(self._early_stop_id)
            patience    = dpg.get_value(self._patience_id)

            # ── Prepare data ──────────────────────────────────────────────
            X = X_df.values.astype(np.float32)
            y = y_series.values.astype(np.float32)

            if task == "Classification":
                classes     = np.unique(y)
                n_classes   = len(classes)
                output_size = 1 if n_classes == 2 else n_classes
                label_map   = {c: i for i, c in enumerate(classes)}
                y           = np.array([label_map[v] for v in y],
                                       dtype=np.float32)
            else:
                output_size = 1
                self._scaler_y = StandardScaler()
                y = self._scaler_y.fit_transform(
                    y.reshape(-1, 1)).flatten()

            self._scaler_X = StandardScaler()
            X = self._scaler_X.fit_transform(X)

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=42)

            X_train_t = torch.tensor(X_train)
            y_train_t = torch.tensor(y_train)
            X_val_t   = torch.tensor(X_val)
            y_val_t   = torch.tensor(y_val)

            # ── Build model ───────────────────────────────────────────────
            if not self._layers:
                self._set_status("Add at least one hidden layer.",
                                 hex_to_rgb("#CC4444"))
                self._is_training = False
                return

            self._model = self._build_model(X.shape[1], output_size)
            optimizer   = self._get_optimizer(self._model)

            if task == "Classification":
                criterion = (nn.BCELoss() if output_size == 1
                             else nn.CrossEntropyLoss())
            else:
                criterion = nn.MSELoss()

            dataset    = torch.utils.data.TensorDataset(X_train_t, y_train_t)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True)

            best_val   = float("inf")
            pat_count  = 0
            history    = {"train_loss": [], "val_loss": []}

            # ── Training loop ─────────────────────────────────────────────
            for epoch in range(epochs):
                self._model.train()
                epoch_loss = 0.0

                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    out = self._model(X_batch)

                    if task == "Classification" and output_size == 1:
                        loss = criterion(out.squeeze(), y_batch)
                    elif task == "Classification":
                        loss = criterion(out, y_batch.long())
                    else:
                        loss = criterion(out.squeeze(), y_batch)

                    if l1_lambda > 0:
                        l1 = sum(p.abs().sum()
                                 for p in self._model.parameters())
                        loss = loss + l1_lambda * l1

                    if l2_lambda > 0:
                        l2 = sum(p.pow(2).sum()
                                 for p in self._model.parameters())
                        loss = loss + l2_lambda * l2

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Validation
                self._model.eval()
                with torch.no_grad():
                    val_out = self._model(X_val_t)
                    if task == "Classification" and output_size == 1:
                        val_loss = criterion(
                            val_out.squeeze(), y_val_t).item()
                    elif task == "Classification":
                        val_loss = criterion(
                            val_out, y_val_t.long()).item()
                    else:
                        val_loss = criterion(
                            val_out.squeeze(), y_val_t).item()

                avg_train = epoch_loss / len(dataloader)
                history["train_loss"].append(avg_train)
                history["val_loss"].append(val_loss)

                dpg.set_value(self._progress_id, (epoch + 1) / epochs)
                self._set_status(
                    f"Epoch {epoch+1}/{epochs}  "
                    f"train={avg_train:.4f}  val={val_loss:.4f}",
                    hex_to_rgb("#2266AA"),
                )

                if early_stop:
                    if val_loss < best_val:
                        best_val  = val_loss
                        pat_count = 0
                    else:
                        pat_count += 1
                        if pat_count >= patience:
                            self._set_status(
                                f"Early stop at epoch {epoch+1}",
                                hex_to_rgb("#CC8800"))
                            break

            # ── Final evaluation ──────────────────────────────────────────
            self._model.eval()
            with torch.no_grad():
                preds_t = self._model(torch.tensor(X))

            if task == "Classification":
                if output_size == 1:
                    preds = (preds_t.squeeze().numpy() > 0.5).astype(int)
                else:
                    preds = preds_t.argmax(dim=1).numpy()
                metric_val  = accuracy_score(y.astype(int), preds)
                metric_name = "accuracy"
            else:
                preds = self._scaler_y.inverse_transform(
                    preds_t.squeeze().numpy().reshape(-1, 1)).flatten()
                y_orig = self._scaler_y.inverse_transform(
                    y.reshape(-1, 1)).flatten()
                metric_val  = r2_score(y_orig, preds)
                metric_name = "r2_score"

            self._result = {
                "predictions": preds,
                "metrics":     {metric_name: metric_val,
                                "history":   history},
                "model":       self._model,
            }
            self._set_status(
                f"Done!  {metric_name} = {metric_val:.4f}",
                hex_to_rgb("#2A7A2A"))

        except Exception as e:
            self._set_status(f"Error: {e}", hex_to_rgb("#CC4444"))
        finally:
            self._is_training = False

    # ══════════════════════════════════════════════════════════════════════
    #  EXECUTE
    # ══════════════════════════════════════════════════════════════════════
    def execute(self, X=None, y=None):
        self._last_X = X
        self._last_y = y
        if self._result is None:
            self._set_status("Hit TRAIN to train the model.",
                             hex_to_rgb("#888888"))
            return {"predictions": None, "metrics": None, "model": None}
        return self._result

    # ══════════════════════════════════════════════════════════════════════
    #  THEME HELPER
    # ══════════════════════════════════════════════════════════════════════
    def _apply_btn_theme(self, btn_id, color):
        darker  = tuple(max(v - 25, 0) for v in color)
        darkest = tuple(max(v - 50, 0) for v in color)
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,
                                    color,   category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,
                                    darker,  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,
                                    darkest, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text,
                                    hex_to_rgb("#FFFFFF"),
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6,
                                    category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme(btn_id, t)

class NodeGraph:
    """Tracks all live nodes and links; runs the graph on demand."""

    def __init__(self):
        self._nodes:        dict[int, BaseNode]       = {}
        self._links:        dict[int, tuple[int, int]] = {}
        self._attr_to_node: dict[int, int]            = {}

    # ── Registration ───────────────────────────────────────────────────────
    def add_node(self, node: BaseNode):
        nid = node.node_id
        self._nodes[nid] = node
        for attr_id in node.input_attrs.values():
            self._attr_to_node[attr_id] = nid
        if node.output_attr is not None:
            self._attr_to_node[node.output_attr] = nid
        for attr_id in node.output_attrs.values():
            self._attr_to_node[attr_id] = nid

    def add_link(self, link_id: int, attr_a: int, attr_b: int):
        out_attr, in_attr = self._classify(attr_a, attr_b)
        self._links[link_id] = (out_attr, in_attr)

    def remove_link(self, link_id: int):
        self._links.pop(link_id, None)

    def _classify(self, a: int, b: int) -> tuple[int, int]:
        """Return (output_attr, input_attr) by checking which side is an output pin."""
        for node in self._nodes.values():
            all_outputs = list(node.output_attrs.values())
            if node.output_attr is not None:
                all_outputs.append(node.output_attr)
            if a in all_outputs: return a, b
            if b in all_outputs: return b, a
        return a, b

    # ── Execution ──────────────────────────────────────────────────────────
    def run(self):
        if not self._nodes:
            print("[Run] No nodes in graph.")
            return

        # input_attr -> output_attr (value flow)
        input_to_source = {in_a: out_a for _, (out_a, in_a) in self._links.items()}

        # Build dependency map
        deps: dict[int, set[int]] = {nid: set() for nid in self._nodes}
        for in_a, out_a in input_to_source.items():
            consumer = self._attr_to_node.get(in_a)
            producer = self._attr_to_node.get(out_a)
            if consumer and producer and consumer != producer:
                deps[consumer].add(producer)

        # Topological sort (DFS)
        visited, order = set(), []

        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            for dep in deps[nid]:
                visit(dep)
            order.append(nid)

        for nid in self._nodes:
            visit(nid)

        # Execute in order
        output_values: dict[int, object] = {}

        print("\n─── Running graph ───")
        for nid in order:
            node   = self._nodes[nid]
            kwargs = {}

            for param_name, in_attr in node.input_attrs.items():
                source_out      = input_to_source.get(in_attr)
                kwargs[param_name] = output_values.get(source_out) if source_out else None

            try:
                result = node.execute(**kwargs)
            except Exception as e:
                result = f"ERROR: {e}"

            # Single output
            if node.output_attr is not None:
                output_values[node.output_attr] = result

            # Multi output — execute() returns a dict keyed by output name
            for name, attr_id in node.output_attrs.items():
                if isinstance(result, dict) and name in result:
                    output_values[attr_id] = result[name]

            args_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"  [{node.LABEL}({args_str})] → {result}")

        print("─────────────────────\n")

#NodeEditor App 
class NodeEditorApp:
    """Owns the DPG viewport and window. Delegates logic to NodeGraph."""

    NODE_PALETTE: list[tuple[str, type[BaseNode]]] = [
        ("Terminal",    TerminalNode),
        ("CSV Loader", CSVNode),
        ("Column Selector", ColumnSelectorNode),
        ("ANN", ANNNode),
    ]

    EDITOR_TAG = "node_editor"

    def __init__(self):
        self.graph    = NodeGraph()
        self._menu_id: int | None = None   # set during _build_ui

    # ── Node factory ───────────────────────────────────────────────────────
    def _spawn(self, node_cls: type[BaseNode], pos: tuple = (10, 10)):
        node = node_cls()
        # Give nodes a graph reference if they need it
        if hasattr(node, 'set_graph'):
            node.set_graph(self.graph)
        node.build(parent=self.EDITOR_TAG, pos=pos)
        self.graph.add_node(node)

    # ── DPG callbacks ──────────────────────────────────────────────────────
    def _on_link(self, sender, app_data):
        link_id = dpg.add_node_link(app_data[0], app_data[1], parent=sender)
        self.graph.add_link(link_id, app_data[0], app_data[1])

    def _on_delink(self, sender, app_data):
        dpg.delete_item(app_data)
        self.graph.remove_link(app_data)

    def _on_right_click(self, sender, app_data):
        if dpg.is_item_hovered(self.EDITOR_TAG):
            pos = dpg.get_mouse_pos(local=False)
            # use integer ID stored on self — string tags can resolve
            # to 0 (not found) on some DPG versions, crashing set_item_pos.
            dpg.set_item_pos(self._menu_id, pos)
            dpg.configure_item(self._menu_id, show=True)

    # DPG calls callbacks as (sender, app_data, user_data) when
    # user_data is set. Using user_data to pass the class is reliable;
    # default-arg lambdas receive user_data as a 3rd positional arg which
    # overwrites the default, making c=cls resolve to None.
    def _on_menu_spawn(self, sender, app_data, user_data):
        self._spawn(user_data)

    def _on_context_spawn(self, sender, app_data, user_data):
        self._spawn(user_data, pos=dpg.get_item_pos(self._menu_id))

    # ── UI build ───────────────────────────────────────────────────────────
    def _build_ui(self):
        with dpg.window(label="Nodeflow", no_title_bar=True, no_resize=True, no_move=True) as self._window_id:
            with dpg.menu_bar():
                with dpg.menu(label="Add Node"):
                    for label, cls in self.NODE_PALETTE:
                        dpg.add_menu_item(
                            label=label,
                            callback=self._on_menu_spawn,
                            user_data=cls,
                        )

            dpg.add_button(label="▶  Run Graph",
                           callback=lambda: self.graph.run(),
                           width=-1)
            dpg.add_separator()

            dpg.add_node_editor(
                tag=self.EDITOR_TAG,
                callback=self._on_link,
                delink_callback=self._on_delink,
            )
        # In _build_ui(), inside your existing handler_registry block:
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=self._on_right_click)
            dpg.add_key_press_handler(key=dpg.mvKey_Delete, callback=self._delete_selected)
            
        # Context menu — stored as int ID, never looked up by string tag
        with dpg.window(show=False, popup=True,
                        no_title_bar=False, min_size=[1, 1]) as self._menu_id:
            for label, cls in self.NODE_PALETTE:
                dpg.add_menu_item(
                    label=f"Add '{label}'",
                    callback=self._on_context_spawn,
                    user_data=cls,
                )

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=self._on_right_click)
            dpg.add_key_press_handler(key=dpg.mvKey_Escape, callback=self._on_escape)
            dpg.add_key_press_handler(key=dpg.mvKey_F11,    callback=self._toggle_fullscreen)
            
    def _on_escape(self):
        if dpg.is_viewport_fullscreen():
            dpg.toggle_viewport_fullscreen()

    def _toggle_fullscreen(self, *args):
        dpg.toggle_viewport_fullscreen()
        self._fit_window_to_viewport()

    def _fit_window_to_viewport(self):
        w = dpg.get_viewport_client_width()
        h = dpg.get_viewport_client_height()
        dpg.set_item_width(self._window_id,  w)
        dpg.set_item_height(self._window_id, h)
        dpg.set_item_pos(self._window_id, [0, 0])
        
    # ── Entry point ────────────────────────────────────────────────────────
    def run(self):
        dpg.create_context()
        NodeEditorTheme.apply_global()
        self._build_ui()
        dpg.create_viewport(
            title="Nodeflow",
            width=1280,
            height=720,
            min_width=800,
            min_height=600,
        )
        dpg.setup_dearpygui()
        dpg.set_primary_window(self._window_id, True)
        dpg.show_viewport(maximized=True)   # ← maximized but keeps OS chrome
        dpg.start_dearpygui()
        dpg.destroy_context()

    def _delete_selected(self):
        selected_nodes = dpg.get_selected_nodes(self.EDITOR_TAG)
        selected_links = dpg.get_selected_links(self.EDITOR_TAG)

        # Delete selected links first
        for link_id in selected_links:
            dpg.delete_item(link_id)
            self.graph.remove_link(link_id)

        # Delete selected nodes + clean up their links
        for node_id in selected_nodes:
            # Remove any links connected to this node before deleting it
            connected = [
                lid for lid, (out_a, in_a) in self.graph._links.items()
                if self.graph._attr_to_node.get(out_a) == node_id
                or self.graph._attr_to_node.get(in_a)  == node_id
            ]
            for lid in connected:
                dpg.delete_item(lid)
                self.graph.remove_link(lid)

            # Remove from graph registry
            node = self.graph._nodes.pop(node_id, None)
            if node:
                if node.output_attr is not None:
                    self.graph._attr_to_node.pop(node.output_attr, None)
                for attr_id in node.input_attrs.values():
                    self.graph._attr_to_node.pop(attr_id, None)
                for attr_id in node.output_attrs.values():
                    self.graph._attr_to_node.pop(attr_id, None)

            dpg.delete_item(node_id)

if __name__ == "__main__":
    NodeEditorApp().run()