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
import json

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

class MakeCSVNode(BaseNode):
    LABEL       = "Make CSV"
    TITLE_COLOR = (45, 95, 160, 255)
    WIDTH       = 280

    def __init__(self):
        super().__init__()
        self._status_id      = None
        self._filename_id    = None
        self._col_name_id    = None
        self._include_idx_id = None
        self._preview_id     = None
        self._save_btn_id    = None
        self._last_data      = {}   # stores all connected inputs for saving

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Input pins ────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("predictions")
            self.input_attrs["predictions"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("y_test (optional)")
            self.input_attrs["y_test (optional)"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("X_test (optional)")
            self.input_attrs["X_test (optional)"] = a

            # ── Config ────────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)

                with dpg.group(horizontal=True):
                    dpg.add_text("Pred col name:", color=hex_to_rgb("#555555"))
                    self._col_name_id = dpg.add_input_text(
                        default_value="predictions",
                        width=130,
                    )

                dpg.add_spacer(height=4)
                self._include_idx_id = dpg.add_checkbox(
                    label="Include row index",
                    default_value=False,
                )
                dpg.add_spacer(height=6)

                # Preview listbox
                dpg.add_text("Preview (first 5 rows):",
                             color=hex_to_rgb("#333333"))
                self._preview_id = dpg.add_listbox(
                    items=[],
                    width=self.WIDTH,
                    num_items=5,
                )
                dpg.add_spacer(height=6)

                self._status_id = dpg.add_text(
                    "Run graph to preview",
                    color=hex_to_rgb("#888888"),
                )
                dpg.add_spacer(height=4)

                save_btn = dpg.add_button(
                    label="Save as CSV…",
                    width=self.WIDTH,
                    height=36,
                    callback=self._open_save_dialog,
                )
                self._apply_btn_theme(save_btn, hex_to_rgb("#2A5A2A"))

            self.output_attr  = None
            self.output_attrs = {}

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    # ── Execution ─────────────────────────────────────────────────────────
    def execute(self, predictions=None, **kwargs):
        y_test = kwargs.get("y_test (optional)")
        X_test = kwargs.get("X_test (optional)")

        if predictions is None:
            dpg.set_value(self._status_id, "Connect predictions pin.")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#888888"))
            dpg.configure_item(self._preview_id, items=[])
            return None

        try:
            df = self._build_dataframe(predictions, y_test, X_test)
            self._last_data = {
                "predictions": predictions,
                "y_test":      y_test,
                "X_test":      X_test,
            }

            # Preview first 5 rows
            header = " | ".join(df.columns.astype(str))
            rows   = [
                " | ".join(
                    f"{v:.4f}" if isinstance(v, float) else str(v)
                    for v in row
                )
                for row in df.head(5).values
            ]
            dpg.configure_item(self._preview_id,
                               items=[f"COLS: {header}", "─"*30] + rows)

            dpg.set_value(self._status_id,
                f"{len(df)} rows  ×  {len(df.columns)} cols  — ready to save")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2266AA"))

        except Exception as e:
            dpg.set_value(self._status_id, f"Error: {e}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))

        return None

    def _build_dataframe(self, predictions, y_test, X_test):
        col_name = dpg.get_value(self._col_name_id) or "predictions"
        preds    = np.array(predictions).flatten()
        df       = pd.DataFrame({col_name: preds})

        # Add y_test column if connected
        if y_test is not None:
            yt = np.array(y_test).flatten()
            # Align lengths
            n = min(len(preds), len(yt))
            df = df.iloc[:n].copy()
            df["y_true"]  = yt[:n]
            df["residual"] = df["y_true"] - df[col_name]

        # Add X_test columns if connected
        if X_test is not None:
            X_arr = np.array(X_test)
            n     = min(len(df), len(X_arr))
            df    = df.iloc[:n].copy()
            if X_arr.ndim == 1:
                df["feature_0"] = X_arr[:n]
            else:
                for i in range(X_arr.shape[1]):
                    df[f"feature_{i}"] = X_arr[:n, i]

        if dpg.get_value(self._include_idx_id):
            df.insert(0, "index", range(len(df)))

        return df

    # ── Save dialog ───────────────────────────────────────────────────────
    def _open_save_dialog(self):
        if not self._last_data:
            dpg.set_value(self._status_id, "Run graph first.")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))
            return
        with dpg.file_dialog(
            label="Save CSV",
            width=500, height=350,
            show=True,
            callback=self._on_save,
            default_filename="predictions.csv",
        ):
            dpg.add_file_extension(".csv", color=(0, 255, 120, 255))
            dpg.add_file_extension(".*",   color=(200, 200, 200, 255))

    def _on_save(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if not path:
            return
        try:
            df = self._build_dataframe(
                self._last_data["predictions"],
                self._last_data.get("y_test"),
                self._last_data.get("X_test"),
            )
            if not path.endswith(".csv"):
                path += ".csv"
            df.to_csv(path, index=False)
            fname = path.split("\\")[-1].split("/")[-1]
            dpg.set_value(self._status_id, f"Saved → {fname}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2A7A2A"))
        except Exception as e:
            dpg.set_value(self._status_id, f"Save error: {e}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))

    # ── Theme helper ──────────────────────────────────────────────────────
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

class TrainTestSplitNode(BaseNode):
    LABEL       = "Train Test Split"
    TITLE_COLOR = (120, 70, 160, 255)
    WIDTH       = 260

    def __init__(self):
        super().__init__()
        self._test_size_id = None
        self._seed_id      = None
        self._stratify_id  = None
        self._status_id    = None

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Input pins ────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("X")
            self.input_attrs["X"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("y")
            self.input_attrs["y"] = a

            # ── Config ────────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)

                with dpg.group(horizontal=True):
                    dpg.add_text("Test Size:  ", color=hex_to_rgb("#555555"))
                    self._test_size_id = dpg.add_slider_float(
                        default_value=0.2,
                        min_value=0.05,
                        max_value=0.5,
                        format="%.2f",
                        width=self.WIDTH - 90,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Random Seed:", color=hex_to_rgb("#555555"))
                    self._seed_id = dpg.add_input_int(
                        default_value=42,
                        width=80,
                    )

                self._stratify_id = dpg.add_checkbox(
                    label="Stratify (classification)",
                    default_value=False,
                )

                dpg.add_spacer(height=4)
                self._status_id = dpg.add_text(
                    "Not split yet", color=hex_to_rgb("#888888"))

            # ── Output pins ───────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as a:
                dpg.add_text("X_train")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as b:
                dpg.add_text("X_test")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as c:
                dpg.add_text("y_train")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as d:
                dpg.add_text("y_test")

            self.output_attrs = {
                "X_train": a, "X_test": b,
                "y_train": c, "y_test": d,
            }
            self.output_attr = None

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    def execute(self, X=None, y=None):
        if X is None or y is None:
            return {"X_train": None, "X_test": None,
                    "y_train": None, "y_test":  None}
        try:
            test_size = dpg.get_value(self._test_size_id)
            seed      = dpg.get_value(self._seed_id)
            stratify  = dpg.get_value(self._stratify_id)

            strat_arr = y if stratify else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=seed,
                stratify=strat_arr,
            )

            dpg.set_value(self._status_id,
                f"Train: {len(X_train)}  Test: {len(X_test)}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2A7A2A"))

            return {
                "X_train": X_train, "X_test":  X_test,
                "y_train": y_train, "y_test":  y_test,
            }
        except Exception as e:
            dpg.set_value(self._status_id, f"Error: {e}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))
            return {"X_train": None, "X_test": None,
                    "y_train": None, "y_test":  None}

class TemporalSplitNode(BaseNode):
    LABEL       = "Temporal Split"
    TITLE_COLOR = (120, 60, 180, 255)
    WIDTH       = 260

    def __init__(self):
        super().__init__()
        self._split_mode_id  = None
        self._test_size_id   = None
        self._test_rows_id   = None
        self._status_id      = None
        self._info_id        = None

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Input pins ────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("X")
            self.input_attrs["X"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("y")
            self.input_attrs["y"] = a

            # ── Config ────────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)

                dpg.add_text("Split Mode:", color=hex_to_rgb("#333333"))
                self._split_mode_id = dpg.add_combo(
                    items=["By Fraction", "By Row Count"],
                    default_value="By Fraction",
                    width=self.WIDTH,
                    callback=self._on_mode_change,
                )
                dpg.add_spacer(height=6)

                # Fraction mode
                with dpg.group() as self._fraction_group:
                    with dpg.group(horizontal=True):
                        dpg.add_text("Test Size:  ", color=hex_to_rgb("#555555"))
                        self._test_size_id = dpg.add_slider_float(
                            default_value=0.2,
                            min_value=0.05,
                            max_value=0.5,
                            format="%.2f",
                            width=self.WIDTH - 90,
                        )

                # Row count mode — hidden by default
                with dpg.group(show=False) as self._rowcount_group:
                    with dpg.group(horizontal=True):
                        dpg.add_text("Test Rows:  ", color=hex_to_rgb("#555555"))
                        self._test_rows_id = dpg.add_input_int(
                            default_value=96,
                            min_value=1,
                            width=110,
                        )
                    dpg.add_text("e.g. 96 = 1 day of 15min data",
                                 color=hex_to_rgb("#888888"))

                dpg.add_spacer(height=6)
                self._info_id = dpg.add_text(
                    "", color=hex_to_rgb("#555555"))
                self._status_id = dpg.add_text(
                    "Not split yet", color=hex_to_rgb("#888888"))

            # ── Output pins ───────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as a:
                dpg.add_text("X_train")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as b:
                dpg.add_text("X_test")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as c:
                dpg.add_text("y_train")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as d:
                dpg.add_text("y_test")

            self.output_attrs = {
                "X_train": a, "X_test": b,
                "y_train": c, "y_test": d,
            }
            self.output_attr = None

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    def _on_mode_change(self):
        mode = dpg.get_value(self._split_mode_id)
        dpg.configure_item(self._fraction_group,
                           show=(mode == "By Fraction"))
        dpg.configure_item(self._rowcount_group,
                           show=(mode == "By Row Count"))

    def execute(self, X=None, y=None):
        if X is None or y is None:
            dpg.set_value(self._status_id, "Connect X and y.")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#888888"))
            return {"X_train": None, "X_test": None,
                    "y_train": None, "y_test":  None}
        try:
            import pandas as pd
            X_arr = np.array(X)
            y_arr = np.array(y)
            n     = len(X_arr)

            mode = dpg.get_value(self._split_mode_id)
            if mode == "By Fraction":
                test_size  = dpg.get_value(self._test_size_id)
                test_rows  = int(n * test_size)
            else:
                test_rows  = dpg.get_value(self._test_rows_id)

            train_rows = n - test_rows

            if train_rows <= 0 or test_rows <= 0:
                dpg.set_value(self._status_id, "Invalid split — adjust size.")
                dpg.configure_item(self._status_id,
                                   color=hex_to_rgb("#CC4444"))
                return {"X_train": None, "X_test": None,
                        "y_train": None, "y_test":  None}

            # Strict temporal split — no shuffling
            X_train = X_arr[:train_rows]
            X_test  = X_arr[train_rows:]
            y_train = y_arr[:train_rows]
            y_test  = y_arr[train_rows:]

            dpg.set_value(self._info_id,
                f"Total: {n}  |  Train: {train_rows}  |  Test: {test_rows}")
            dpg.configure_item(self._info_id, color=hex_to_rgb("#555555"))
            dpg.set_value(self._status_id,
                f"Split done — {train_rows/n*100:.1f}% / "
                f"{test_rows/n*100:.1f}%")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2A7A2A"))

            return {
                "X_train": X_train, "X_test":  X_test,
                "y_train": y_train, "y_test":  y_test,
            }

        except Exception as e:
            dpg.set_value(self._status_id, f"Error: {e}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))
            return {"X_train": None, "X_test": None,
                    "y_train": None, "y_test":  None}

class ScalerNode(BaseNode):
    LABEL       = "Scaler"
    TITLE_COLOR = (160, 100, 30, 255)
    WIDTH       = 240

    def __init__(self):
        super().__init__()
        self._scaler_type_id = None
        self._status_id      = None
        self._scaler         = None

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Input pin ─────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("X")
            self.input_attrs["X"] = a

            # ── Config ────────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)
                dpg.add_text("Scaler Type:", color=hex_to_rgb("#333333"))
                self._scaler_type_id = dpg.add_combo(
                    items=[
                        "StandardScaler",
                        "MinMaxScaler",
                        "RobustScaler",
                        "MaxAbsScaler",
                        "Normalizer",
                    ],
                    default_value="StandardScaler",
                    width=self.WIDTH,
                )
                dpg.add_spacer(height=4)
                self._status_id = dpg.add_text(
                    "Not scaled yet", color=hex_to_rgb("#888888"))

            # ── Output pins ───────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as a:
                dpg.add_text("X_scaled")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as b:
                dpg.add_text("scaler")   # pass scaler downstream for inverse_transform

            self.output_attrs = {"X_scaled": a, "scaler": b}
            self.output_attr  = None

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    def _get_scaler(self, name):
        from sklearn.preprocessing import (
            StandardScaler, MinMaxScaler,
            RobustScaler, MaxAbsScaler, Normalizer,
        )
        return {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler":   MinMaxScaler(),
            "RobustScaler":   RobustScaler(),
            "MaxAbsScaler":   MaxAbsScaler(),
            "Normalizer":     Normalizer(),
        }.get(name, StandardScaler())

    def execute(self, X=None):
        if X is None:
            return {"X_scaled": None, "scaler": None}
        try:
            import pandas as pd
            name         = dpg.get_value(self._scaler_type_id)
            self._scaler = self._get_scaler(name)

            # Handle Series — reshape to 2D, scale, flatten back to 1D
            is_series = isinstance(X, pd.Series)
            arr       = np.array(X).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            scaled = self._scaler.fit_transform(arr)

            # Flatten back if input was 1D
            if is_series or scaled.shape[1] == 1:
                scaled = scaled.flatten()

            dpg.set_value(self._status_id,
            f"{name} applied  shape={scaled.shape}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2A7A2A"))

            return {"X_scaled": scaled, "scaler": self._scaler}

        except Exception as e:
            dpg.set_value(self._status_id, f"Error: {e}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))
            return {"X_scaled": None, "scaler": None}

class InverseScalerNode(BaseNode):
    LABEL       = "Inverse Scaler"
    TITLE_COLOR = (160, 100, 30, 255)
    WIDTH       = 240

    def __init__(self):
        super().__init__()
        self._status_id = None

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("X_scaled")
            self.input_attrs["X_scaled"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("scaler")
            self.input_attrs["scaler"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)
                self._status_id = dpg.add_text(
                    "Connect X_scaled and scaler",
                    color=hex_to_rgb("#888888"),
                )

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as a:
                dpg.add_text("X_original")
            self.output_attrs = {"X_original": a}
            self.output_attr  = None

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    def execute(self, X_scaled=None, scaler=None):
        if X_scaled is None:
            dpg.set_value(self._status_id, "X_scaled not connected.")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))
            return {"X_original": None}

        if scaler is None:
            dpg.set_value(self._status_id, "Scaler not connected.")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))
            return {"X_original": None}

        try:
            arr = np.array(X_scaled).astype(np.float32)
            is_1d = arr.ndim == 1
            if is_1d:
                arr = arr.reshape(-1, 1)

            result = scaler.inverse_transform(arr)

            if is_1d:
                result = result.flatten()

            dpg.set_value(self._status_id,
                f"Restored  shape={result.shape}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2A7A2A"))

            return {"X_original": result}

        except Exception as e:
            dpg.set_value(self._status_id, f"Error: {e}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))
            return {"X_original": None}

class InferenceNode(BaseNode):
    LABEL       = "Inference"
    TITLE_COLOR = (160, 40, 60, 255)
    WIDTH       = 300

    def __init__(self):
        super().__init__()
        self._status_id     = None
        self._info_id       = None
        self._model         = None
        self._scaler        = None
        self._device        = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Input pins ────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("X")
            self.input_attrs["X"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("scaler (optional)")
            self.input_attrs["scaler (optional)"] = a

            # ── Body ──────────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)

                # Load model button
                load_btn = dpg.add_button(
                    label="Load Model (.pt)",
                    width=self.WIDTH,
                    height=36,
                    callback=self._load_model,
                )
                self._apply_btn_theme(load_btn, hex_to_rgb("#2A2A6A"))
                dpg.add_spacer(height=6)

                self._info_id = dpg.add_text(
                    "No model loaded",
                    color=hex_to_rgb("#888888"),
                )
                dpg.add_spacer(height=4)
                self._status_id = dpg.add_text(
                    "Load a model and connect X",
                    color=hex_to_rgb("#888888"),
                )

            # ── Output pins ───────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as a:
                dpg.add_text("predictions")
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as b:
                dpg.add_text("predictions_raw")

            self.output_attrs = {
                "predictions":     a,   # inverse transformed if scaler connected
                "predictions_raw": b,   # raw model output always
            }
            self.output_attr = None

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    # ── Model loading ─────────────────────────────────────────────────────
    def _load_model(self):
        with dpg.file_dialog(
            label="Load Model",
            width=500, height=350,
            show=True,
            callback=self._on_model_loaded,
        ):
            dpg.add_file_extension(".pt",  color=(0, 255, 120, 255))
            dpg.add_file_extension(".*",   color=(200, 200, 200, 255))

    def _on_model_loaded(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if not path:
            return
        try:
            checkpoint = torch.load(
                path, map_location=self._device, weights_only=False)

            # Handle both raw model and our checkpoint dict format
            if isinstance(checkpoint, dict):
                self._model = checkpoint.get("model")
                # Also grab scalers if saved with our ANN node format
                self._scaler_X = checkpoint.get("scaler_X")
                self._scaler_y = checkpoint.get("scaler_y")
            else:
                # Raw model object saved directly
                self._model    = checkpoint
                self._scaler_X = None
                self._scaler_y = None

            if self._model is None:
                dpg.set_value(self._info_id, "No model found in file.")
                dpg.configure_item(self._info_id, color=hex_to_rgb("#CC4444"))
                return

            self._model = self._model.to(self._device)
            self._model.eval()

            # Show model summary
            fname    = path.split("\\")[-1].split("/")[-1]
            n_params = sum(p.numel() for p in self._model.parameters())
            dpg.set_value(self._info_id,
                f"{fname}\n{n_params:,} parameters  [{self._device}]")
            dpg.configure_item(self._info_id, color=hex_to_rgb("#2A7A2A"))
            dpg.set_value(self._status_id, "Model ready — connect X and run graph")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2266AA"))

        except Exception as e:
            dpg.set_value(self._info_id, f"Load error: {e}")
            dpg.configure_item(self._info_id, color=hex_to_rgb("#CC4444"))

    # ── Execution ─────────────────────────────────────────────────────────
    def execute(self, X=None, **kwargs):
        # Grab optional scaler from kwargs
        scaler_in = kwargs.get("scaler (optional)")

        if self._model is None:
            dpg.set_value(self._status_id, "Load a model first.")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))
            return {"predictions": None, "predictions_raw": None}

        if X is None:
            dpg.set_value(self._status_id, "Connect X input.")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))
            return {"predictions": None, "predictions_raw": None}

        try:
            # ── Prepare input ─────────────────────────────────────────────
            X_arr = np.array(X).astype(np.float32)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)  # single sample

            # Apply X scaler if available from checkpoint
            if self._scaler_X is not None:
                X_arr = self._scaler_X.transform(X_arr).astype(np.float32)
            elif scaler_in is not None:
                # Use externally connected scaler
                arr = X_arr
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                X_arr = scaler_in.transform(arr).astype(np.float32)

            # ── Run inference ─────────────────────────────────────────────
            self._model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_arr).to(self._device)
                out      = self._model(X_tensor)
                preds_raw = out.cpu().numpy()

            # ── Inverse transform if y scaler available ───────────────────
            if self._scaler_y is not None:
                p = preds_raw.reshape(-1, 1)
                preds = self._scaler_y.inverse_transform(p).flatten()
            else:
                preds = preds_raw.flatten()

            dpg.set_value(self._status_id,
                f"Done  {len(preds)} prediction(s)  [{self._device}]")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2A7A2A"))

            return {
                "predictions":     preds,
                "predictions_raw": preds_raw.flatten(),
            }

        except Exception as e:
            dpg.set_value(self._status_id, f"Error: {e}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))
            return {"predictions": None, "predictions_raw": None}

    # ── Theme helper ──────────────────────────────────────────────────────
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
        self._task_id        = None
        self._output_size_id = None
        self._optimizer_id   = None
        self._lr_id          = None
        self._epochs_id      = None
        self._batch_id       = None
        self._val_split_id   = None

        # ── Regularization config ─────────────────────────────────────────
        self._dropout_id    = None
        self._l1_id         = None
        self._l2_id         = None
        self._wd_id         = None
        self._early_stop_id = None
        self._patience_id   = None

        # ── Status / progress ─────────────────────────────────────────────
        self._status_id   = None
        self._progress_id = None

        # ── Runtime ───────────────────────────────────────────────────────
        self._model       = None
        self._scaler_X    = None
        self._scaler_y    = None
        self._is_training = False
        self._result      = None
        self._last_X      = None
        self._last_y      = None
        self._last_X_test = None
        self._device      = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("X_test")
            self.input_attrs["X_test"] = a

            # ── Body ──────────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)

                # Manual tab buttons — no tab_bar to avoid border bleed
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

                with dpg.group(show=True) as g:
                    self._build_layers_tab()
                self._tab_groups["Layers"] = g

                with dpg.group(show=False) as g:
                    self._build_training_tab()
                self._tab_groups["Training"] = g

                with dpg.group(show=False) as g:
                    self._build_regularization_tab()
                self._tab_groups["Regularization"] = g

                dpg.add_spacer(height=8)

                self._status_id = dpg.add_text(
                    "Ready", color=hex_to_rgb("#555555"))
                self._progress_id = dpg.add_progress_bar(
                    default_value=0.0, width=self.WIDTH)
                dpg.add_spacer(height=4)

                train_btn = dpg.add_button(
                    label="TRAIN",
                    width=self.WIDTH,
                    height=38,
                    callback=self._on_train_click,
                )
                self._apply_btn_theme(train_btn, hex_to_rgb("#2D6A9F"))

                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    save_btn = dpg.add_button(
                        label="Save Model",
                        width=self.WIDTH // 2 - 2,
                        callback=self._save_model,
                    )
                    load_btn = dpg.add_button(
                        label="Load Model",
                        width=self.WIDTH // 2 - 2,
                        callback=self._load_model,
                    )
                    self._apply_btn_theme(save_btn, hex_to_rgb("#2A5A2A"))
                    self._apply_btn_theme(load_btn, hex_to_rgb("#2A2A6A"))

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
    #  TAB CONTENT — plain widgets, no dpg.tab() wrapper
    # ══════════════════════════════════════════════════════════════════════
    def _build_layers_tab(self):
        dpg.add_text("Task:", color=hex_to_rgb("#333333"))
        self._task_id = dpg.add_combo(
            items=["Classification", "Regression"],
            default_value="Classification",
            width=self.WIDTH,
        )
        dpg.add_spacer(height=6)

        with dpg.group(horizontal=True):
            dpg.add_text("Output Neurons:", color=hex_to_rgb("#555555"))
            self._output_size_id = dpg.add_input_int(
                default_value=1,
                min_value=1,
                max_value=4096,
                width=90,
            )
        dpg.add_spacer(height=2)
        dpg.add_text("Define ALL layers including output layer",
                     color=hex_to_rgb("#888888"))
        dpg.add_spacer(height=6)

        dpg.add_text("Layers  (input → ... → output):",
                     color=hex_to_rgb("#333333"))
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
                items=["ReLU", "Sigmoid", "Tanh",
                       "LeakyReLU", "ELU", "GELU", "None"],
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

    def _build_model(self, input_size):
        """Build model exactly as defined in the layer list.
        No automatic output layer — user defines everything."""
        dropout = dpg.get_value(self._dropout_id)
        layers  = []
        prev    = input_size

        for layer in self._layers:
            layers.append(nn.Linear(prev, layer["units"]))
            layers.append(self._get_activation(layer["activation"]))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = layer["units"]

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

    def _train(self, X_raw, y_raw):
        self._is_training = True
        self._set_status("Preparing data...", hex_to_rgb("#2266AA"))

        try:
            task        = dpg.get_value(self._task_id)
            output_size = dpg.get_value(self._output_size_id)
            epochs      = dpg.get_value(self._epochs_id)
            batch_size  = dpg.get_value(self._batch_id)
            val_split   = dpg.get_value(self._val_split_id)
            l1_lambda   = dpg.get_value(self._l1_id)
            l2_lambda   = dpg.get_value(self._l2_id)
            early_stop  = dpg.get_value(self._early_stop_id)
            patience    = dpg.get_value(self._patience_id)

            # ── Prepare data ──────────────────────────────────────────────
            # np.array() handles both DataFrames and numpy arrays safely
            X = np.array(X_raw).astype(np.float32)
            y = np.array(y_raw).astype(np.float32)

            if task == "Classification":
                classes   = np.unique(y)
                label_map = {c: i for i, c in enumerate(classes)}
                y         = np.array([label_map[v] for v in y],
                                     dtype=np.float32)
            else:
                self._scaler_y = StandardScaler()
                y = self._scaler_y.fit_transform(
                    y.reshape(-1, 1)).flatten().astype(np.float32)

            self._scaler_X = StandardScaler()
            X = self._scaler_X.fit_transform(X).astype(np.float32)

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=42)

            X_train_t = torch.tensor(X_train).to(self._device)
            y_train_t = torch.tensor(y_train).to(self._device)
            X_val_t   = torch.tensor(X_val).to(self._device)
            y_val_t   = torch.tensor(y_val).to(self._device)

            # ── Build model ───────────────────────────────────────────────
            if not self._layers:
                self._set_status("Add at least one layer.",
                                 hex_to_rgb("#CC4444"))
                self._is_training = False
                return

            self._model = self._build_model(X.shape[1]).to(self._device)
            optimizer   = self._get_optimizer(self._model)

            if task == "Classification":
                criterion = (nn.BCELoss() if output_size == 1
                             else nn.CrossEntropyLoss())
            else:
                criterion = nn.MSELoss()

            dataset    = torch.utils.data.TensorDataset(X_train_t, y_train_t)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True)

            best_val  = float("inf")
            pat_count = 0
            history   = {"train_loss": [], "val_loss": []}

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
                        l1   = sum(p.abs().sum()
                                   for p in self._model.parameters())
                        loss = loss + l1_lambda * l1

                    if l2_lambda > 0:
                        l2   = sum(p.pow(2).sum()
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

            # ── Generate predictions on test set ──────────────────────────
            self._model.eval()
            with torch.no_grad():
                if self._last_X_test is not None:
                    X_eval = self._scaler_X.transform(
                        np.array(self._last_X_test).astype(np.float32))
                else:
                    X_eval = X   # fallback to training data

                preds_t = self._model(
                    torch.tensor(X_eval).to(self._device))

            if task == "Classification":
                if output_size == 1:
                    preds = (preds_t.cpu().squeeze().numpy()
                             > 0.5).astype(int)
                else:
                    preds = preds_t.cpu().argmax(dim=1).numpy()
                metric_val  = None   # compute via MetricsNode
                metric_name = "accuracy"
            else:
                preds = self._scaler_y.inverse_transform(
                    preds_t.cpu().squeeze().numpy().reshape(
                        -1, 1)).flatten()
                metric_val  = None   # compute via MetricsNode
                metric_name = "r2_score"

            self._result = {
                "predictions": preds,
                "metrics":     {"history": history},
                "model":       self._model,
            }
            self._set_status(
                f"Done! Predictions ready  [{self._device}]",
                hex_to_rgb("#2A7A2A"),
            )

        except Exception as e:
            self._set_status(f"Error: {e}", hex_to_rgb("#CC4444"))
        finally:
            self._is_training = False

    # ══════════════════════════════════════════════════════════════════════
    #  EXECUTE
    # ══════════════════════════════════════════════════════════════════════
    def execute(self, X=None, y=None, X_test=None):
        self._last_X      = X
        self._last_y      = y
        self._last_X_test = X_test

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

    # Graph Saviing
    
    def _save_model(self):
        if self._model is None:
            self._set_status("No model to save — train first.",
                             hex_to_rgb("#CC4444"))
            return
        with dpg.file_dialog(
            label="Save Model",
            width=500, height=350,
            show=True,
            callback=self._on_save_model,
            default_filename="model.pt",
        ):
            dpg.add_file_extension(".pt",  color=(0, 255, 120, 255))
            dpg.add_file_extension(".*",   color=(200, 200, 200, 255))

    def _on_save_model(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if not path:
            return
        try:
            torch.save({
                "model_state":  self._model.state_dict(),
                "model":        self._model,
                "scaler_X":     self._scaler_X,
                "scaler_y":     self._scaler_y,
                "layers":       self._layers,
            }, path)
            self._set_status(f"Saved → {path.split('/')[-1]}",
                             hex_to_rgb("#2A7A2A"))
        except Exception as e:
            self._set_status(f"Save error: {e}", hex_to_rgb("#CC4444"))

    def _load_model(self):
        with dpg.file_dialog(
            label="Load Model",
            width=500, height=350,
            show=True,
            callback=self._on_load_model,
        ):
            dpg.add_file_extension(".pt",  color=(0, 255, 120, 255))
            dpg.add_file_extension(".*",   color=(200, 200, 200, 255))

    def _on_load_model(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if not path:
            return
        try:
            checkpoint      = torch.load(path, map_location=self._device ,weights_only=False)
            self._model     = checkpoint["model"].to(self._device)
            self._scaler_X  = checkpoint.get("scaler_X")
            self._scaler_y  = checkpoint.get("scaler_y")
            self._layers    = checkpoint.get("layers", [])

            # Restore layer list display
            dpg.configure_item(self._layer_list_id,
                               items=[self._layer_str(l)
                                      for l in self._layers])

            self._result = {
                "predictions": None,
                "metrics":     None,
                "model":       self._model,
            }
            self._set_status(f"Loaded ← {path.split('/')[-1]}",
                             hex_to_rgb("#2A7A2A"))
        except Exception as e:
            self._set_status(f"Load error: {e}", hex_to_rgb("#CC4444"))
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
#Custom Visualization Nodes:
class ScatterPlotNode(BaseNode):
    LABEL       = "Scatter Plot"
    TITLE_COLOR = (40, 140, 120, 255)
    WIDTH       = 300
    HEIGHT      = 300

    def __init__(self):
        super().__init__()
        self._plot_id    = None
        self._series_id  = None
        self._line_id    = None
        self._status_id  = None

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Input pins ────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("y_test")
            self.input_attrs["y_test"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("predictions")
            self.input_attrs["predictions"] = a

            # ── Plot ──────────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)
                self._status_id = dpg.add_text(
                    "Connect y_test and predictions",
                    color=hex_to_rgb("#888888"),
                )
                dpg.add_spacer(height=4)

                with dpg.plot(
                    label="Actual vs Predicted",
                    width=self.WIDTH,
                    height=self.HEIGHT,
                    no_title=False,
                ) as self._plot_id:
                    dpg.add_plot_legend()
                    x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Actual")
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Predicted")

                    # Scatter series — starts empty
                    self._series_id = dpg.add_scatter_series(
                        [], [],
                        label="Predictions",
                        parent=y_axis,
                    )
                    # Perfect prediction line y=x
                    self._line_id = dpg.add_line_series(
                        [], [],
                        label="Perfect fit",
                        parent=y_axis,
                    )

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    def execute(self, y_test=None, predictions=None):
        if y_test is None or predictions is None:
            dpg.set_value(self._status_id, "Waiting for data...")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#888888"))
            return None

        try:
            # Convert to plain python lists for DPG
            y_true = list(map(float, y_test))
            y_pred = list(map(float, predictions))

            # Update scatter
            dpg.set_value(self._series_id, [y_true, y_pred])

            # Update perfect fit line across data range
            mn = min(y_true)
            mx = max(y_true)
            dpg.set_value(self._line_id, [[mn, mx], [mn, mx]])

            # Fit axes to data
            dpg.fit_axis_data(dpg.get_item_parent(self._series_id))
            dpg.fit_axis_data(dpg.get_item_parent(self._line_id))

            # Compute R2 for status
            ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
            mean_y = sum(y_true) / len(y_true)
            ss_tot = sum((a - mean_y) ** 2 for a in y_true)
            r2     = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

            dpg.set_value(self._status_id, f"R² = {r2:.4f}  n={len(y_true)}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2A7A2A"))

        except Exception as e:
            dpg.set_value(self._status_id, f"Error: {e}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))

        return None

class DataInspectorNode(BaseNode):
    LABEL       = "Data Inspector"
    TITLE_COLOR = (60, 60, 60, 255)
    WIDTH       = 320

    def __init__(self):
        super().__init__()
        self._info_id    = None
        self._list_id    = None
        self._label_id   = None

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("data")
            self.input_attrs["data"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)
                self._label_id = dpg.add_text(
                    "No data", color=hex_to_rgb("#888888"))
                self._info_id  = dpg.add_text(
                    "", color=hex_to_rgb("#555555"))
                dpg.add_spacer(height=4)
                self._list_id = dpg.add_listbox(
                    items=[],
                    width=self.WIDTH,
                    num_items=10,
                )

            self.output_attr  = None
            self.output_attrs = {}

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    def execute(self, data=None):
        if data is None:
            dpg.set_value(self._label_id, "No data received")
            dpg.configure_item(self._label_id, color=hex_to_rgb("#CC4444"))
            dpg.configure_item(self._list_id, items=[])
            return None

        import pandas as pd
        data = np.array(data) if not isinstance(data, (pd.DataFrame, pd.Series)) else data

        # ── Info line ─────────────────────────────────────────────────────
        if isinstance(data, pd.DataFrame):
            dtype_str = "DataFrame"
            shape_str = f"{data.shape[0]} rows × {data.shape[1]} cols"
            rows = [" | ".join(str(v) for v in row)
                    for row in data.head(20).values]
            header = " | ".join(data.columns.astype(str))
            items  = [f"COLUMNS: {header}", "─" * 40] + rows

        elif isinstance(data, pd.Series):
            dtype_str = "Series"
            shape_str = f"{len(data)} values  dtype={data.dtype}"
            items = [str(v) for v in data.head(20).values]

        else:
            arr = np.array(data)
            dtype_str = "Array"
            shape_str = f"shape={arr.shape}  dtype={arr.dtype}"
            if arr.ndim == 1:
                items = [str(v) for v in arr[:20]]
            else:
                items = [" | ".join(f"{v:.4f}" if isinstance(v, float)
                                    else str(v) for v in row)
                         for row in arr[:20]]

        dpg.set_value(self._label_id, dtype_str)
        dpg.configure_item(self._label_id, color=hex_to_rgb("#2266AA"))
        dpg.set_value(self._info_id,  shape_str)
        dpg.configure_item(self._list_id, items=items)
        return None

class NetworkVisualizerNode(BaseNode):
    LABEL       = "Network Visualizer"
    TITLE_COLOR = (80, 60, 150, 255)
    WIDTH       = 340
    HEIGHT      = 280

    # Visual constants
    NEURON_R    = 10
    LAYER_GAP   = 70
    MAX_NEURONS = 6    # max neurons to draw per layer before collapsing

    def __init__(self):
        super().__init__()
        self._canvas_id  = None
        self._status_id  = None
        self._last_model = None

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Input pin ─────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("model")
            self.input_attrs["model"] = a

            # ── Body ──────────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)

                self._status_id = dpg.add_text(
                    "Connect model pin then hit Refresh",
                    color=hex_to_rgb("#888888"),
                )
                dpg.add_spacer(height=4)

                refresh_btn = dpg.add_button(
                    label="↺ Refresh",
                    width=self.WIDTH,
                    callback=self._refresh,
                )
                self._apply_btn_theme(refresh_btn, hex_to_rgb("#4A4A8A"))
                dpg.add_spacer(height=6)

                # Drawlist canvas
                self._canvas_id = dpg.add_drawlist(
                    width=self.WIDTH,
                    height=self.HEIGHT,
                )
                self._draw_placeholder()

            self.output_attr  = None
            self.output_attrs = {}

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    # ── Drawing ───────────────────────────────────────────────────────────
    def _draw_placeholder(self):
        dpg.draw_text(
            (self.WIDTH // 2 - 60, self.HEIGHT // 2 - 8),
            "No model loaded",
            color=hex_to_rgb("#666666"),
            size=14,
            parent=self._canvas_id,
        )

    def _clear_canvas(self):
        dpg.delete_item(self._canvas_id, children_only=True)

    def _draw_network(self, layer_sizes: list[int], layer_labels: list[str]):
        self._clear_canvas()

        n_layers  = len(layer_sizes)
        total_w   = self.WIDTH
        total_h   = self.HEIGHT
        r         = self.NEURON_R
        pad_x     = 30
        pad_y     = 20

        # X positions for each layer
        if n_layers == 1:
            x_positions = [total_w // 2]
        else:
            step = (total_w - 2 * pad_x) / (n_layers - 1)
            x_positions = [int(pad_x + i * step) for i in range(n_layers)]

        # Neuron Y positions per layer
        def neuron_ys(count):
            display = min(count, self.MAX_NEURONS)
            if display == 1:
                return [total_h // 2], count > 1
            gap = (total_h - 2 * pad_y) / (display - 1)
            ys  = [int(pad_y + j * gap) for j in range(display)]
            return ys, count > self.MAX_NEURONS

        layer_ys = []
        truncated = []
        for size in layer_sizes:
            ys, trunc = neuron_ys(size)
            layer_ys.append(ys)
            truncated.append(trunc)

        # Draw connections first (behind neurons)
        for i in range(n_layers - 1):
            for y1 in layer_ys[i]:
                for y2 in layer_ys[i + 1]:
                    dpg.draw_line(
                        (x_positions[i],     y1),
                        (x_positions[i + 1], y2),
                        color=hex_to_rgb("#555577", alpha=120),
                        thickness=1,
                        parent=self._canvas_id,
                    )

        # Draw neurons
        for i, (x, ys) in enumerate(zip(x_positions, layer_ys)):
            is_input  = (i == 0)
            is_output = (i == n_layers - 1)

            if is_input:
                color    = hex_to_rgb("#4A9A6A")   # green — input
                outline  = hex_to_rgb("#2A7A4A")
            elif is_output:
                color    = hex_to_rgb("#9A4A4A")   # red — output
                outline  = hex_to_rgb("#7A2A2A")
            else:
                color    = hex_to_rgb("#4A6A9A")   # blue — hidden
                outline  = hex_to_rgb("#2A4A7A")

            for y in ys:
                dpg.draw_circle(
                    (x, y), r,
                    color=outline,
                    fill=color,
                    thickness=1.5,
                    parent=self._canvas_id,
                )

            # Draw "..." if truncated
            if truncated[i]:
                last_y = ys[-1]
                dpg.draw_text(
                    (x - 6, last_y + r + 4),
                    "...",
                    color=hex_to_rgb("#AAAAAA"),
                    size=12,
                    parent=self._canvas_id,
                )

            # Layer label below
            dpg.draw_text(
                (x - 14, total_h - 26),
                layer_labels[i].split("\n")[0],   # "in" / "L" / "out"
                color=hex_to_rgb("#AAAAAA"),
                size=11,
                parent=self._canvas_id,
            )
            # Exact neuron count below
            dpg.draw_text(
                (x - 10, total_h - 14),
                str(layer_sizes[i]),              # exact number
                color=hex_to_rgb("#DDDDDD"),
                size=11,
                parent=self._canvas_id,
            )

    def _parse_model(self, model):
        """Extract layer sizes and labels from a PyTorch Sequential model."""
        import torch.nn as nn
        layer_sizes  = []
        layer_labels = []

        # Input size from first Linear layer
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layer_sizes.append(module.in_features)
                layer_labels.append(f"in\n{module.in_features}")
                break

        # Hidden + output layers
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layer_sizes.append(module.out_features)
                layer_labels.append(f"L\n{module.out_features}")

        # Fix last label
        if layer_labels:
            layer_labels[-1] = f"out\n{layer_sizes[-1]}"

        return layer_sizes, layer_labels

    # ── Refresh ───────────────────────────────────────────────────────────
    def _refresh(self):
        if self._last_model is None:
            dpg.set_value(self._status_id,
                          "No model yet — run graph first.")
            dpg.configure_item(self._status_id,
                               color=hex_to_rgb("#CC4444"))
            return
        try:
            sizes, labels = self._parse_model(self._last_model)
            self._draw_network(sizes, labels)
            arch = " → ".join(str(s) for s in sizes)
            dpg.set_value(self._status_id, f"{arch}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2A7A2A"))
        except Exception as e:
            dpg.set_value(self._status_id, f"Error: {e}")
            dpg.configure_item(self._status_id,
                               color=hex_to_rgb("#CC4444"))

    # ── Execute ───────────────────────────────────────────────────────────
    def execute(self, model=None):
        if model is not None:
            self._last_model = model
            dpg.set_value(self._status_id,
                          "Model received — hit ↺ Refresh")
            dpg.configure_item(self._status_id,
                               color=hex_to_rgb("#2266AA"))
        return None

    # ── Theme helper ──────────────────────────────────────────────────────
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

class MetricsNode(BaseNode):
    LABEL       = "Test Metrics"
    TITLE_COLOR = (160, 120, 30, 255)
    WIDTH       = 280

    METRICS = ["R²", "RMSE", "MAE", "MAPE", "MSE", "Accuracy", "F1", "Precision", "Recall"]

    def __init__(self):
        super().__init__()
        self._checks    = {}   # metric name -> checkbox id
        self._results   = {}   # metric name -> text widget id
        self._status_id = None

    def build(self, parent, pos=(10, 10)):
        with dpg.node(label=self.LABEL, parent=parent, pos=pos) as self.node_id:

            # ── Input pins ────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("y_true")
            self.input_attrs["y_true"] = a

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as a:
                dpg.add_text("predictions")
            self.input_attrs["predictions"] = a

            # ── Body ──────────────────────────────────────────────────────
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_spacer(height=4)
                dpg.add_text("Select Metrics:", color=hex_to_rgb("#333333"))
                dpg.add_spacer(height=4)

                # Checkboxes — two columns
                for i in range(0, len(self.METRICS), 2):
                    with dpg.group(horizontal=True):
                        m1 = self.METRICS[i]
                        self._checks[m1] = dpg.add_checkbox(
                            label=m1,
                            default_value=True if m1 in ["R²", "RMSE", "MAE"] else False,
                        )
                        if i + 1 < len(self.METRICS):
                            dpg.add_spacer(width=16)
                            m2 = self.METRICS[i + 1]
                            self._checks[m2] = dpg.add_checkbox(
                                label=m2,
                                default_value=False,
                            )

                dpg.add_spacer(height=8)
                dpg.add_spacer(height=6)

                # Results display
                dpg.add_text("Results:", color=hex_to_rgb("#333333"))
                dpg.add_spacer(height=4)
                for metric in self.METRICS:
                    self._results[metric] = dpg.add_text(
                        f"{metric}: —",
                        color=hex_to_rgb("#888888"),
                        show=False,
                    )

                dpg.add_spacer(height=4)
                self._status_id = dpg.add_text(
                    "Run graph to compute metrics",
                    color=hex_to_rgb("#888888"),
                )

            self.output_attr  = None
            self.output_attrs = {}

        NodeEditorTheme.apply_to_node(self.node_id, self.TITLE_COLOR)
        return self.node_id

    # ── Execution ─────────────────────────────────────────────────────────
    def execute(self, y_true=None, predictions=None):
        if y_true is None or predictions is None:
            dpg.set_value(self._status_id, "Waiting for data...")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#888888"))
            return None

        try:
            yt = np.array(y_true,       dtype=np.float64).flatten()
            yp = np.array(predictions,  dtype=np.float64).flatten()

            # Hide all result rows first
            for metric in self.METRICS:
                dpg.configure_item(self._results[metric], show=False)

            computed = {}

            for metric in self.METRICS:
                if not dpg.get_value(self._checks[metric]):
                    continue

                val = self._compute(metric, yt, yp)
                computed[metric] = val

                label = f"{metric}: {val}" if isinstance(val, str) else f"{metric}: {val:.6f}"
                dpg.set_value(self._results[metric], label)
                dpg.configure_item(
                    self._results[metric],
                    color=hex_to_rgb("#2A7A2A"),
                    show=True,
                )

            dpg.set_value(self._status_id,
                          f"{len(computed)} metric(s) computed")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#2266AA"))

        except Exception as e:
            dpg.set_value(self._status_id, f"Error: {e}")
            dpg.configure_item(self._status_id, color=hex_to_rgb("#CC4444"))

        return None

    def _compute(self, metric, yt, yp):
        from sklearn.metrics import (
            r2_score, mean_squared_error, mean_absolute_error,
            accuracy_score, f1_score, precision_score, recall_score,
        )
        try:
            if metric == "R²":
                return r2_score(yt, yp)
            elif metric == "RMSE":
                return np.sqrt(mean_squared_error(yt, yp))
            elif metric == "MAE":
                return mean_absolute_error(yt, yp)
            elif metric == "MSE":
                return mean_squared_error(yt, yp)
            elif metric == "MAPE":
                mask = yt != 0
                return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)
            elif metric == "Accuracy":
                return accuracy_score(yt.astype(int), yp.astype(int))
            elif metric == "F1":
                return f1_score(yt.astype(int), yp.astype(int), average="weighted", zero_division=0)
            elif metric == "Precision":
                return precision_score(yt.astype(int), yp.astype(int), average="weighted", zero_division=0)
            elif metric == "Recall":
                return recall_score(yt.astype(int), yp.astype(int), average="weighted", zero_division=0)
        except Exception as e:
            return f"Error: {e}"
#NodeEditor App 
class NodeEditorApp:
    """Owns the DPG viewport and window. Delegates logic to NodeGraph."""

    NODE_PALETTE: list[tuple[str, type[BaseNode]]] = [
        ("Terminal",TerminalNode),
        ("CSV Loader", CSVNode),
        ("Column Selector",ColumnSelectorNode),
        ("ANN", ANNNode),
        ("Train Test Split",TrainTestSplitNode),
        ("Scaler",ScalerNode),
        ("Scatter Plot",ScatterPlotNode),
        ("Network Visualizer",NetworkVisualizerNode),
        ("Test Metrics", MetricsNode),
        ("Data Inspector", DataInspectorNode),
        ("Inverse Scaler", InverseScalerNode),
        ("Temporal Split", TemporalSplitNode),
        ("Inference", InferenceNode),
        ("Make CSV", MakeCSVNode),
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

                with dpg.menu(label="Graph"):
                    dpg.add_menu_item(label="Save Graph…", callback=self._save_graph)
                    dpg.add_menu_item(label="Load Graph…", callback=self._load_graph)

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
            
    def _save_graph(self):
        with dpg.file_dialog(
            label="Save Graph",
            width=500, height=350,
            show=True,
            callback=self._on_save_graph,
            default_filename="graph.json",
        ):
            dpg.add_file_extension(".json", color=(0, 200, 255, 255))

    def _on_save_graph(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if path:
            GraphSerializer.save(self, path)
            print(f"[Graph] Saved → {path}")

    def _load_graph(self):
        with dpg.file_dialog(
            label="Load Graph",
            width=500, height=350,
            show=True,
            callback=self._on_load_graph,
        ):
            dpg.add_file_extension(".json", color=(0, 200, 255, 255))

    def _on_load_graph(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if path:
            GraphSerializer.load(self, path)
            print(f"[Graph] Loaded ← {path}")

class GraphSerializer:
    """Saves and loads node graph layout to/from JSON."""

    @staticmethod
    def save(app: "NodeEditorApp", path: str):
        data = {"nodes": [], "links": []}

        for nid, node in app.graph._nodes.items():
            pos = dpg.get_item_pos(nid)
            data["nodes"].append({
                "type":  type(node).__name__,
                "pos":   pos,
            })

        for lid, (out_a, in_a) in app.graph._links.items():
            # Find which node each attr belongs to
            out_nid = app.graph._attr_to_node.get(out_a)
            in_nid  = app.graph._attr_to_node.get(in_a)
            if out_nid is None or in_nid is None:
                continue

            out_node = app.graph._nodes[out_nid]
            in_node  = app.graph._nodes[in_nid]

            # Find pin names
            out_pin = next(
                (k for k, v in out_node.output_attrs.items() if v == out_a),
                "output"
            )
            in_pin = next(
                (k for k, v in in_node.input_attrs.items() if v == in_a),
                "input"
            )

            data["links"].append({
                "from_node": type(out_node).__name__,
                "from_pos":  dpg.get_item_pos(out_nid),
                "from_pin":  out_pin,
                "to_node":   type(in_node).__name__,
                "to_pos":    dpg.get_item_pos(in_nid),
                "to_pin":    in_pin,
            })

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(app: "NodeEditorApp", path: str):
        with open(path) as f:
            data = json.load(f)

        # Build name → class map from palette
        cls_map = {cls.__name__: cls
                   for _, cls in app.NODE_PALETTE}

        # Spawn nodes — track pos → node for link resolution
        pos_to_node: dict[str, BaseNode] = {}

        for entry in data["nodes"]:
            cls = cls_map.get(entry["type"])
            if cls is None:
                print(f"[Load] Unknown node type: {entry['type']}")
                continue
            pos   = tuple(entry["pos"])
            node  = cls()
            if hasattr(node, "set_graph"):
                node.set_graph(app.graph)
            node.build(parent=app.EDITOR_TAG, pos=pos)
            app.graph.add_node(node)
            key = f"{entry['type']}@{pos[0]},{pos[1]}"
            pos_to_node[key] = node

        # Re-create links
        for link in data["links"]:
            from_key = (f"{link['from_node']}@"
                        f"{link['from_pos'][0]},{link['from_pos'][1]}")
            to_key   = (f"{link['to_node']}@"
                        f"{link['to_pos'][0]},{link['to_pos'][1]}")

            from_node = pos_to_node.get(from_key)
            to_node   = pos_to_node.get(to_key)

            if from_node is None or to_node is None:
                continue

            out_attr = from_node.output_attrs.get(link["from_pin"])
            in_attr  = to_node.input_attrs.get(link["to_pin"])

            if out_attr is None or in_attr is None:
                continue

            link_id = dpg.add_node_link(
                out_attr, in_attr,
                parent=app.EDITOR_TAG,
            )
            app.graph.add_link(link_id, out_attr, in_attr)

if __name__ == "__main__":
    NodeEditorApp().run()