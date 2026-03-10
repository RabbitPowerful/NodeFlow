import inspect
import random
from abc import ABC, abstractmethod

import dearpygui.dearpygui as dpg

import pandas as pd
# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════

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
                cls._c("mvNodeCol_GridBackground",     (20,  20,  30, 255))
                cls._c("mvNodeCol_GridLine",           (60,  60,  80, 120))
                cls._c("mvNodeCol_Link",               (100, 200, 255, 200))
                cls._c("mvNodeCol_LinkHovered",        (150, 230, 255, 255))
                cls._c("mvNodeCol_LinkSelected",       (255, 220, 100, 255))
                cls._c("mvNodeCol_BoxSelector",        (100, 180, 255,  50))
                cls._c("mvNodeCol_BoxSelectorOutline", (100, 180, 255, 180))
                cls._s("mvNodesStyleVar_GridSpacing",              24)
                cls._s("mvNodesStyleVar_LinkThickness",             2)
                cls._s("mvNodesStyleVar_LinkLineSegmentsPerLength", 0.1)
                cls._s("mvNodesStyleVar_LinkHoverDistance",         10)
                cls._s("mvNodesStyleVar_PinCircleRadius",           5)
                cls._s("mvNodesStyleVar_PinLineThickness",          2)
                cls._s("mvNodesStyleVar_PinHoverRadius",            10)
                cls._s("mvNodesStyleVar_PinOffset",                 0)

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


# ══════════════════════════════════════════════════════════════════════════════
#  BASE NODE
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  CONCRETE NODES  — add new ones here or in separate files
# ══════════════════════════════════════════════════════════════════════════════

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
                dpg.add_separator()

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

# ══════════════════════════════════════════════════════════════════════════════
#  NODE GRAPH  — owns registries and execution logic
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════════

class NodeEditorApp:
    """Owns the DPG viewport and window. Delegates logic to NodeGraph."""

    NODE_PALETTE: list[tuple[str, type[BaseNode]]] = [
        ("Terminal",    TerminalNode),
        ("CSV Loader", CSVNode),
        ("Column Selector", ColumnSelectorNode),
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
            # FIX 1: use integer ID stored on self — string tags can resolve
            # to 0 (not found) on some DPG versions, crashing set_item_pos.
            dpg.set_item_pos(self._menu_id, pos)
            dpg.configure_item(self._menu_id, show=True)

    # FIX 2: DPG calls callbacks as (sender, app_data, user_data) when
    # user_data is set. Using user_data to pass the class is reliable;
    # default-arg lambdas receive user_data as a 3rd positional arg which
    # overwrites the default, making c=cls resolve to None.
    def _on_menu_spawn(self, sender, app_data, user_data):
        self._spawn(user_data)

    def _on_context_spawn(self, sender, app_data, user_data):
        self._spawn(user_data, pos=dpg.get_item_pos(self._menu_id))

    # ── UI build ───────────────────────────────────────────────────────────
    def _build_ui(self):
        with dpg.window(label="Node Editor", width=800, height=600):
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

        # Context menu — stored as int ID, never looked up by string tag
        with dpg.window(show=False, popup=True,
                        no_title_bar=True, min_size=[1, 1]) as self._menu_id:
            for label, cls in self.NODE_PALETTE:
                dpg.add_menu_item(
                    label=f"Add '{label}'",
                    callback=self._on_context_spawn,
                    user_data=cls,
                )

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right,
                callback=self._on_right_click,
            )

    # ── Entry point ────────────────────────────────────────────────────────
    def run(self):
        dpg.create_context()
        NodeEditorTheme.apply_global()
        self._build_ui()
        dpg.create_viewport(title="Modular Node Editor", width=1000, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    NodeEditorApp().run()