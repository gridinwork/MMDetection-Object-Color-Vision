"""Right-hand controls: modes, presets, classes, and object info."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.object_info_panel import ObjectInfoPanel

PRESETS = {
    "detection": {
        "detection": True,
        "segmentation": False,
        "tracking": False,
        "color": False,
        "trails": False,
        "labels": True,
        "confidence": True,
        "object_id": False,
        "debug": False,
        "detailed_color": False,
        "palette": False,
    },
    "segmentation": {
        "detection": True,
        "segmentation": True,
        "tracking": False,
        "color": False,
        "trails": False,
        "labels": True,
        "confidence": True,
        "object_id": False,
        "detailed_color": False,
        "palette": False,
    },
    "tracking": {
        "detection": True,
        "segmentation": False,
        "tracking": True,
        "color": False,
        "trails": False,
        "labels": True,
        "confidence": True,
        "object_id": True,
        "detailed_color": False,
        "palette": False,
    },
    "color": {
        "detection": True,
        "segmentation": False,
        "tracking": False,
        "color": True,
        "trails": False,
        "labels": True,
        "confidence": True,
        "object_id": False,
        "detailed_color": False,
        "palette": False,
    },
    "all": {
        "detection": True,
        "segmentation": True,
        "tracking": True,
        "color": True,
        "trails": True,
        "labels": True,
        "confidence": True,
        "object_id": True,
        "debug": False,
        "detailed_color": False,
        "palette": True,
    },
}


class ControlPanel(QWidget):
    settings_changed = Signal()
    lock_requested = Signal()
    save_object_requested = Signal()
    save_masked_requested = Signal()
    object_selected = Signal(object)

    def __init__(self):
        super().__init__()
        self._updating = False
        self._class_names: list[str] = []
        self._object_keys: tuple = ()
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(self._modes_box())
        root.addWidget(self._preset_box())
        root.addWidget(self._confidence_box())
        root.addWidget(self._class_box())
        root.addWidget(self._objects_box())
        root.addWidget(self._info_box())
        root.addWidget(self._gpu_box())
        self.debug_box = self._debug_box()
        root.addWidget(self.debug_box)
        root.addStretch(1)
        self.info.lock_button.clicked.connect(self.lock_requested.emit)
        self.info.save_button.clicked.connect(self.save_object_requested.emit)
        self.info.mask_button.clicked.connect(self.save_masked_requested.emit)

    def _modes_box(self) -> QGroupBox:
        box = QGroupBox("DETECTION MODES")
        layout = QVBoxLayout(box)
        self.checks: dict[str, QCheckBox] = {}
        rows = [
            ("detection", "Object Detection"),
            ("segmentation", "Instance Segmentation"),
            ("tracking", "Object Tracking"),
            ("color", "Color Analysis"),
            ("trails", "Motion Trails"),
            ("labels", "Labels"),
            ("confidence", "Confidence"),
            ("object_id", "Object ID"),
            ("detailed_color", "Detailed Color Info"),
            ("palette", "Show Color Palette"),
            ("debug", "Debug"),
        ]
        for key, title in rows:
            check = QCheckBox(title)
            check.toggled.connect(self._emit_settings)
            self.checks[key] = check
            layout.addWidget(check)
        return box

    def _preset_box(self) -> QGroupBox:
        box = QGroupBox("PRESETS")
        layout = QHBoxLayout(box)
        for key, title in (
            ("detection", "DETECTION"),
            ("segmentation", "SEGMENTATION"),
            ("tracking", "TRACKING"),
            ("color", "COLOR"),
            ("all", "ALL"),
        ):
            button = QPushButton(title)
            button.setObjectName("presetButton")
            button.clicked.connect(lambda _checked=False, name=key: self.apply_preset(name))
            layout.addWidget(button)
        return box

    def _confidence_box(self) -> QGroupBox:
        box = QGroupBox("DETECTION CONFIDENCE")
        layout = QVBoxLayout(box)
        self.confidence_value = QLabel("0.40")
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(10, 95)
        self.confidence_slider.setValue(40)
        self.confidence_slider.valueChanged.connect(self._on_confidence)
        layout.addWidget(self.confidence_value)
        layout.addWidget(self.confidence_slider)
        return box

    def _class_box(self) -> QGroupBox:
        box = QGroupBox("CLASS FILTER")
        layout = QVBoxLayout(box)
        self.detected_label = QLabel("Detected now: —")
        self.detected_label.setWordWrap(True)
        self.detected_label.setObjectName("muted")
        layout.addWidget(self.detected_label)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search Class")
        self.search.textChanged.connect(self._apply_search)
        layout.addWidget(self.search)
        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(160)
        self.class_list.itemChanged.connect(self._emit_settings)
        layout.addWidget(self.class_list)
        row = QHBoxLayout()
        all_button = QPushButton("ALL")
        none_button = QPushButton("NONE")
        all_button.clicked.connect(lambda: self._set_all_classes(True))
        none_button.clicked.connect(lambda: self._set_all_classes(False))
        row.addWidget(all_button)
        row.addWidget(none_button)
        layout.addLayout(row)
        return box

    def _objects_box(self) -> QGroupBox:
        box = QGroupBox("OBJECTS")
        layout = QVBoxLayout(box)
        self.object_list = QListWidget()
        self.object_list.setMinimumHeight(110)
        self.object_list.currentItemChanged.connect(self._on_object_item)
        layout.addWidget(self.object_list)
        return box

    def _info_box(self) -> QGroupBox:
        box = QGroupBox("OBJECT INFO")
        layout = QVBoxLayout(box)
        self.info = ObjectInfoPanel()
        layout.addWidget(self.info)
        return box

    def _gpu_box(self) -> QGroupBox:
        box = QGroupBox("DEVICE")
        layout = QVBoxLayout(box)
        self.gpu_label = QLabel("GPU: —\nVRAM: —")
        self.gpu_label.setWordWrap(True)
        layout.addWidget(self.gpu_label)
        return box

    def _debug_box(self) -> QGroupBox:
        box = QGroupBox("DEBUG")
        layout = QVBoxLayout(box)
        self.debug_label = QLabel("Debug is off")
        self.debug_label.setWordWrap(True)
        self.debug_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.debug_label)
        box.setVisible(False)
        return box

    def apply_preset(self, name: str) -> None:
        preset = PRESETS[name]
        self._updating = True
        for key, value in preset.items():
            if key in self.checks:
                self.checks[key].setChecked(bool(value))
        self._updating = False
        self.settings_changed.emit()

    def set_modes(self, modes: dict) -> None:
        self._updating = True
        for key, check in self.checks.items():
            if key in modes:
                check.setChecked(bool(modes[key]))
        self._updating = False

    def modes(self) -> dict:
        return {key: check.isChecked() for key, check in self.checks.items()}

    def set_confidence(self, value: float) -> None:
        self._updating = True
        self.confidence_slider.setValue(int(round(float(value) * 100)))
        self.confidence_value.setText(f"{float(value):.2f}")
        self._updating = False

    def confidence(self) -> float:
        return self.confidence_slider.value() / 100.0

    def set_class_names(self, names: list[str], disabled: list[str] | None = None) -> None:
        cleaned = []
        seen = set()
        for name in names:
            text = str(name)
            if text and text not in seen:
                seen.add(text)
                cleaned.append(text)
        if cleaned == self._class_names:
            return
        disabled_set = set(disabled or self.disabled_classes())
        self._class_names = cleaned
        self._updating = True
        self.class_list.clear()
        for name in cleaned:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Unchecked if name in disabled_set else Qt.CheckState.Checked
            )
            self.class_list.addItem(item)
        self._updating = False
        self._apply_search(self.search.text())

    def disabled_classes(self) -> list[str]:
        names = []
        for index in range(self.class_list.count()):
            item = self.class_list.item(index)
            if item.checkState() != Qt.CheckState.Checked:
                names.append(item.text())
        return names

    def set_detected(self, names: list[str]) -> None:
        if names:
            self.detected_label.setText("Detected now: " + ", ".join(names))
        else:
            self.detected_label.setText("Detected now: —")

    def set_objects(self, objects: list[dict], selected_key) -> None:
        visible = [obj for obj in objects if obj.get("class_name") not in set(self.disabled_classes())]
        keys = tuple(obj.get("select_key") for obj in visible)
        self._updating = True
        if keys != self._object_keys:
            self.object_list.clear()
            for obj in visible:
                item = QListWidgetItem(self._object_text(obj))
                item.setData(Qt.ItemDataRole.UserRole, obj.get("select_key"))
                self.object_list.addItem(item)
            self._object_keys = keys
        else:
            for index, obj in enumerate(visible):
                item = self.object_list.item(index)
                if item is not None:
                    item.setText(self._object_text(obj))
        if selected_key is not None:
            for index in range(self.object_list.count()):
                item = self.object_list.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == selected_key:
                    self.object_list.setCurrentItem(item)
                    break
        self._updating = False

    def set_gpu(self, text: str) -> None:
        self.gpu_label.setText(text)

    def set_debug(self, text: str, visible: bool) -> None:
        self.debug_box.setVisible(visible)
        self.debug_label.setText(text if visible else "Debug is off")

    def show_object(self, obj: dict | None, locked: bool) -> None:
        if obj is None:
            self.info.show_empty()
            return
        self.info.show_object(obj, locked)

    def _object_text(self, obj: dict) -> str:
        name = str(obj.get("class_name", "")).upper()
        track = obj.get("track_id")
        prefix = f"#{track} " if track is not None else ""
        color = f"  {obj['color_name']}" if obj.get("color_name") else ""
        return f"{prefix}{name}  {float(obj.get('score', 0.0)) * 100:.0f}%{color}"

    def _on_confidence(self, value: int) -> None:
        self.confidence_value.setText(f"{value / 100.0:.2f}")
        self._emit_settings()

    def _emit_settings(self, *_args) -> None:
        if not self._updating:
            self.settings_changed.emit()

    def _apply_search(self, text: str) -> None:
        query = text.strip().lower()
        for index in range(self.class_list.count()):
            item = self.class_list.item(index)
            item.setHidden(query not in item.text().lower())

    def _set_all_classes(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._updating = True
        for index in range(self.class_list.count()):
            self.class_list.item(index).setCheckState(state)
        self._updating = False
        self.settings_changed.emit()

    def _on_object_item(self, current, _previous) -> None:
        if self._updating or current is None:
            return
        self.object_selected.emit(current.data(Qt.ItemDataRole.UserRole))
