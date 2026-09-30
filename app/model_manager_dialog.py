"""Download, delete, and verify MMDetection checkpoints."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from inference.model_registry import ModelRegistry, ModelSpec
from utils.downloader import download_file
from utils.logger import get_logger

log = get_logger("models")


class _DownloadThread(QThread):
    progress = Signal(int, int)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, url: str, dest):
        super().__init__()
        self.url = url
        self.dest = dest

    def run(self) -> None:
        try:
            download_file(self.url, self.dest, progress=lambda done, total: self.progress.emit(done, total))
            self.succeeded.emit(str(self.dest))
        except Exception as exc:
            log.exception("Model download failed")
            self.failed.emit(str(exc))


class _VerifyThread(QThread):
    finished_text = Signal(bool, str)

    def __init__(self, registry: ModelRegistry, spec: ModelSpec):
        super().__init__()
        self.registry = registry
        self.spec = spec

    def run(self) -> None:
        ok, message = self.registry.verify(self.spec)
        self.finished_text.emit(ok, message)


class ModelManagerDialog(QDialog):
    changed = Signal()

    def __init__(self, registry: ModelRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.setWindowTitle("Model Manager")
        self.resize(820, 460)
        self._download: _DownloadThread | None = None
        self._verify: _VerifyThread | None = None
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Model", "Task", "Profile", "Status", "File"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.message = QLabel("Select a model. After download the studio works offline.")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QHBoxLayout()
        self.download_button = QPushButton("DOWNLOAD")
        self.delete_button = QPushButton("DELETE")
        self.verify_button = QPushButton("VERIFY")
        close_button = QPushButton("CLOSE")
        self.download_button.clicked.connect(self._download_selected)
        self.delete_button.clicked.connect(self._delete_selected)
        self.verify_button.clicked.connect(self._verify_selected)
        close_button.clicked.connect(self.accept)
        buttons.addWidget(self.download_button)
        buttons.addWidget(self.delete_button)
        buttons.addWidget(self.verify_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self) -> None:
        self.registry.reload()
        specs = self.registry.all()
        self.table.setRowCount(len(specs))
        for row, spec in enumerate(specs):
            values = [
                spec.display_name,
                "Instance Segmentation" if spec.supports_masks else "Detection",
                (spec.profile or "—").upper(),
                self.registry.status_text(spec),
                spec.checkpoint_path().name,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(256, spec.model_id)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()

    def _selected(self) -> ModelSpec | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        return self.registry.get(str(item.data(256)))

    def _download_selected(self) -> None:
        spec = self._selected()
        if spec is None:
            self.message.setText("Select a model first.")
            return
        if self.registry.is_installed(spec):
            self.message.setText(f"{spec.display_name} is already installed.")
            return
        if not spec.weights_url:
            self.message.setText("This custom model has no download URL. Place the checkpoint yourself.")
            return
        if self._download is not None and self._download.isRunning():
            self.message.setText("A download is already running.")
            return
        self.progress.setValue(0)
        self.message.setText(f"Downloading {spec.display_name}...")
        self._set_busy(True)
        thread = _DownloadThread(spec.weights_url, spec.checkpoint_path())
        thread.progress.connect(self._on_progress)
        thread.succeeded.connect(self._on_downloaded)
        thread.failed.connect(self._on_failed)
        thread.finished.connect(lambda: self._set_busy(False))
        self._download = thread
        thread.start()

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
            return
        self.progress.setRange(0, 1000)
        self.progress.setValue(int(done / total * 1000))
        self.message.setText(f"Downloading {done / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB")

    def _on_downloaded(self, _path: str) -> None:
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000)
        self.message.setText("Download finished. The model is ready offline.")
        self.refresh()
        self.changed.emit()

    def _on_failed(self, message: str) -> None:
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.message.setText(message)
        QMessageBox.warning(self, "Download", message)

    def _delete_selected(self) -> None:
        spec = self._selected()
        if spec is None:
            return
        answer = QMessageBox.question(self, "Delete", f"Delete the checkpoint for {spec.display_name}?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        ok, message = self.registry.delete_checkpoint(spec)
        self.message.setText(message)
        if not ok:
            QMessageBox.warning(self, "Delete", message)
        self.refresh()
        self.changed.emit()

    def _verify_selected(self) -> None:
        spec = self._selected()
        if spec is None:
            return
        if self._verify is not None and self._verify.isRunning():
            return
        self.message.setText(f"Verifying {spec.display_name}...")
        thread = _VerifyThread(self.registry, spec)
        thread.finished_text.connect(self._on_verified)
        self._verify = thread
        thread.start()

    def _on_verified(self, ok: bool, message: str) -> None:
        self.message.setText(message)
        if not ok:
            QMessageBox.warning(self, "Verify", message)

    def _set_busy(self, busy: bool) -> None:
        self.download_button.setEnabled(not busy)
        self.delete_button.setEnabled(not busy)
        self.verify_button.setEnabled(not busy)
