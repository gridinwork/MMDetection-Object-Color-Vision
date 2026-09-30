"""Map a measured HSV/LAB color to a stable color name."""

from __future__ import annotations

import json

from utils.paths import color_ranges_file

_FALLBACK = {
    "neutral": {"max_saturation": 32, "white_min_value": 185, "black_max_value": 45},
    "dark_value": 100,
    "beige": {"h_min": 8, "h_max": 40, "s_min": 15, "s_max": 95, "v_min": 150},
    "pink": {
        "s_min": 25,
        "s_max": 170,
        "v_min": 170,
        "ranges": [{"h_min": 0, "h_max": 12}, {"h_min": 150, "h_max": 179}],
    },
    "brown": {
        "h_min": 6,
        "h_max": 28,
        "s_min": 45,
        "s_max": 220,
        "v_min": 25,
        "v_max": 165,
        "b_min": 135,
    },
    "hues": [
        {"name": "RED", "ranges": [{"h_min": 0, "h_max": 8}, {"h_min": 170, "h_max": 180}]},
        {"name": "ORANGE", "ranges": [{"h_min": 8, "h_max": 20}]},
        {"name": "YELLOW", "ranges": [{"h_min": 20, "h_max": 38}]},
        {"name": "GREEN", "ranges": [{"h_min": 38, "h_max": 85}]},
        {"name": "CYAN", "ranges": [{"h_min": 85, "h_max": 98}]},
        {"name": "BLUE", "ranges": [{"h_min": 98, "h_max": 130}]},
        {"name": "PURPLE", "ranges": [{"h_min": 130, "h_max": 170}]},
    ],
}


def _hue_match(hue: int, ranges: list[dict]) -> bool:
    for item in ranges:
        if int(item["h_min"]) <= hue <= int(item["h_max"]):
            return True
    return False


class ColorClassifier:
    def __init__(self, ranges: dict | None = None):
        self.ranges = ranges if ranges is not None else self.load_ranges()

    @staticmethod
    def load_ranges() -> dict:
        path = color_ranges_file()
        if not path.exists():
            return json.loads(json.dumps(_FALLBACK))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return json.loads(json.dumps(_FALLBACK))
        if not isinstance(data, dict):
            return json.loads(json.dumps(_FALLBACK))
        return data

    def classify(self, hsv, lab) -> str:
        hue = int(hsv[0])
        saturation = int(hsv[1])
        value = int(hsv[2])
        lab_b = int(lab[2]) if lab is not None and len(lab) > 2 else 128
        neutral = self.ranges.get("neutral", _FALLBACK["neutral"])
        if saturation <= int(neutral["max_saturation"]):
            if value >= int(neutral["white_min_value"]):
                return "WHITE"
            if value <= int(neutral["black_max_value"]):
                return "BLACK"
            return "GRAY"

        beige = self.ranges.get("beige", _FALLBACK["beige"])
        if (
            int(beige["h_min"]) <= hue <= int(beige["h_max"])
            and int(beige["s_min"]) <= saturation <= int(beige["s_max"])
            and value >= int(beige["v_min"])
        ):
            return "BEIGE"

        pink = self.ranges.get("pink", _FALLBACK["pink"])
        pink_max_s = int(pink.get("s_max", 255))
        if (
            int(pink["s_min"]) <= saturation <= pink_max_s
            and value >= int(pink["v_min"])
            and _hue_match(hue, pink.get("ranges", []))
        ):
            return "PINK"

        brown = self.ranges.get("brown", _FALLBACK["brown"])
        if (
            int(brown["h_min"]) <= hue <= int(brown["h_max"])
            and int(brown["s_min"]) <= saturation <= int(brown["s_max"])
            and int(brown["v_min"]) <= value <= int(brown["v_max"])
            and lab_b >= int(brown.get("b_min", 0))
        ):
            return "BROWN"

        name = "GRAY"
        for bucket in self.ranges.get("hues", []):
            if _hue_match(hue, bucket.get("ranges", [])):
                name = str(bucket["name"])
                break

        dark_value = int(self.ranges.get("dark_value", 100))
        if value <= dark_value and name in {"RED", "GREEN", "BLUE"}:
            return f"DARK {name}"
        return name
