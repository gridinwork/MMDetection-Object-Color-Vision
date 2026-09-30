"""Dark studio stylesheet."""

STYLESHEET = """
QWidget {
    background: #1a1d23;
    color: #e7e9ee;
    font-family: "Segoe UI";
    font-size: 13px;
}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {
    background: #1a1d23;
}
QFrame#toolbar {
    background: #22262e;
    border: 1px solid #343a46;
    border-radius: 8px;
}
QFrame#videoBar {
    background: #22262e;
    border: 1px solid #343a46;
    border-radius: 8px;
}
QLabel#sectionTitle {
    color: #9eb6d6;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
}
QLabel#hint, QLabel#pathLabel, QLabel#muted {
    color: #9aa3b2;
}
QGroupBox {
    background: #22262e;
    border: 1px solid #343a46;
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #c5d4ea;
}
QPushButton {
    background: #2d3440;
    border: 1px solid #465062;
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 28px;
}
QPushButton:hover { background: #384254; }
QPushButton:pressed { background: #2a3342; }
QPushButton:disabled { color: #6d7582; background: #262b33; }
QPushButton#startButton { background: #2c7a56; border-color: #3d9b6e; color: white; font-weight: 700; }
QPushButton#startButton:hover { background: #349468; }
QPushButton#stopButton { background: #8a3e3e; border-color: #b15a5a; color: white; font-weight: 700; }
QPushButton#presetButton { padding: 4px 8px; min-height: 26px; }
QComboBox, QLineEdit, QSpinBox {
    background: #2a303a;
    border: 1px solid #465062;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 28px;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #2a303a;
    selection-background-color: #35527a;
    border: 1px solid #465062;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #343b48;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -6px 0;
    background: #4c8dff;
    border-radius: 8px;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #6a7688;
    background: #2a303a;
}
QCheckBox::indicator:checked {
    background: #3d7eef;
    border-color: #3d7eef;
}
QListWidget {
    background: #1c2129;
    border: 1px solid #343a46;
    border-radius: 6px;
}
QListWidget::item { padding: 3px 4px; }
QListWidget::item:selected { background: #35527a; }
QPlainTextEdit, QTextEdit {
    background: #14181e;
    border: 1px solid #343a46;
    border-radius: 6px;
}
QProgressBar {
    background: #2a303a;
    border: 1px solid #465062;
    border-radius: 5px;
    text-align: center;
    min-height: 16px;
}
QProgressBar::chunk { background: #3d7eef; border-radius: 4px; }
QStatusBar { background: #161920; color: #c5ceda; }
QScrollBar:vertical { background: #1a1d23; width: 12px; }
QScrollBar::handle:vertical { background: #3c4452; border-radius: 5px; min-height: 24px; }
QTableWidget { background: #1c2129; gridline-color: #343a46; border: 1px solid #343a46; }
QHeaderView::section { background: #2a303a; color: #d5dbe3; padding: 6px; border: none; }
"""
