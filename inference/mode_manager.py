"""Choose one model for the active mode. Never runs two models at once."""

from __future__ import annotations

from inference.model_registry import ModelRegistry, ModelSpec

_OPT_TO_PROFILE = {
    "max_fps": "fast",
    "balanced": "balanced",
    "max_quality": "accurate",
}


class ModeManager:
    def resolve(
        self,
        registry: ModelRegistry,
        *,
        segmentation: bool,
        profile: str,
        optimization: str,
        explicit_model_id: str | None,
    ) -> ModelSpec:
        task = "instance_segmentation" if segmentation else "detection"
        chosen = (profile or "fast").lower()
        if chosen == "auto":
            chosen = _OPT_TO_PROFILE.get((optimization or "balanced").lower(), "balanced")
        explicit = registry.get(explicit_model_id or "")
        if explicit is not None and self._matches(explicit, task):
            return explicit
        spec = registry.for_profile(task, chosen)
        if spec is not None:
            return spec
        models = registry.by_task(task)
        if models:
            return models[0]
        raise LookupError(f"No model is registered for {task}")

    @staticmethod
    def _matches(spec: ModelSpec, task: str) -> bool:
        if task == "instance_segmentation":
            return spec.supports_masks
        return spec.task == "detection" and not spec.supports_masks
