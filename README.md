# MMDetection Object & Color Vision Studio

A local real-time computer-vision desktop application built around **OpenMMLab MMDetection 3.3.0**, RTMDet / RTMDet-Ins, MMCV, MMEngine, PyTorch, OpenCV and PySide6.

The application combines object detection, instance segmentation, persistent multi-object tracking, motion analysis and pixel-based color recognition in one Windows interface for webcam, video and still-image sources.

## Overview

This project turns the MMDetection inference ecosystem into a practical interactive vision workstation. Instead of separate command-line inference scripts, the user can select a source, model, device and processing profile from a desktop GUI and immediately inspect detections, segmentation masks, tracking IDs, motion trails and object-color information.

MMDetection is used as an external dependency and is not vendored into this repository. The desktop application layer, source management, tracker, color-analysis pipeline, visualization, model manager and workflow integration are implemented in this project.

## Main features

- Real-time object detection
- Instance segmentation
- Persistent object IDs between frames
- ByteTrack-style two-stage association implemented locally
- Motion-direction estimation
- Motion trails / trajectory history
- Pixel-based object color analysis
- RGB, HSV and LAB measurements
- Dominant-color palette extraction
- Per-class filtering
- Adjustable detection confidence
- Object locking and inspection
- Annotated screenshot export
- Clean-frame export
- Cropped-object export
- Transparent masked-object export
- Webcam, video and image sources
- NVIDIA CUDA / CPU execution
- Explicit GPU selection and AUTO device mode
- Runtime FPS and GPU memory information
- Integrated model manager
- Custom MMDetection model registration

## Detection and segmentation

The application supports RTMDet detection models and RTMDet-Ins instance-segmentation models available through the MMDetection model ecosystem. A Mask R-CNN option is also supported in the model registry.

Typical profiles map UI choices to different model sizes:

| Profile | Typical goal |
| --- | --- |
| Fast | Maximum throughput |
| Balanced | General-purpose speed/quality balance |
| Accurate | Higher detector accuracy |

Model checkpoint files are intentionally excluded from GitHub and can be downloaded separately.

## Persistent tracking

Tracking is performed locally without requiring MMTracking. The tracker follows a ByteTrack-style strategy:

1. Match high-confidence detections first.
2. Use lower-confidence detections to recover existing tracks.
3. Keep temporarily lost tracks alive for a configurable window.
4. Store recent center positions for direction and trajectory rendering.

This lets labels such as `PERSON #1` or `CUP #2` remain associated with the same physical object as it moves through the scene.

## Motion analysis

Recent object-center positions are used to estimate simple states such as:

- STATIC
- MOVING LEFT
- MOVING RIGHT
- MOVING UP
- MOVING DOWN

Optional motion trails visualize recent positions directly over the frame.

## Pixel-based color analysis

Color is calculated from the image pixels belonging to a detected object, not from the class name.

When a segmentation mask is available, the analyzer can sample only pixels inside the object mask. Otherwise it uses a reduced region inside the bounding box to decrease background contamination.

The color subsystem calculates:

- mean RGB
- HSV
- LAB
- named color classification
- dominant color palette and proportions

Reference ranges are stored in `color/color_ranges.json`.

## Supported sources

### Webcam
Live camera processing with selectable camera, resolution, model and hardware device.

### Video
Supported formats include MP4, AVI, MOV and MKV. Playback supports pause, stop and timeline interaction while the inference pipeline stays near real time by avoiding an unbounded frame queue.

### Image
JPG, JPEG, PNG and WEBP images can be loaded for immediate detection/segmentation analysis.

## GPU / CPU execution

Device modes include:

- `AUTO` — prefer the most suitable available CUDA device, otherwise CPU
- explicit `CUDA:N`
- CPU

The interface can display runtime model/device information and GPU memory usage where available.

## Model Manager

The integrated Model Manager can:

- show installed / not-installed status
- download checkpoints
- verify PyTorch checkpoint files
- remove local checkpoints
- switch between detection and segmentation models

Large third-party checkpoint files are not committed to this repository.

## Custom models

The architecture supports registration of custom MMDetection models without rewriting the main application. Custom registry entries can define:

- local or package config
- checkpoint path
- task type
- class definitions
- mask support
- profile metadata

See `custom_models/README.md`.

## Installation

Recommended environment:

- Windows 10/11
- Python 3.11
- NVIDIA GPU optional but recommended

Run:

```bat
install.bat
```

Then launch:

```bat
start.bat
```

The verified dependency stack includes PyTorch, MMCV, MMEngine, MMDetection, PySide6, OpenCV and NumPy.

## Repository structure

```text
app/              PySide6 desktop interface and pipeline orchestration
capture/          camera, video and image sources
color/            color analysis and dominant-color extraction
inference/        MMDetection backend and model registry
tracking/         persistent object tracking and trajectories
visualization/    boxes, masks, labels and rendering
utils/            GPU, downloader, settings and helper utilities
custom_models/    custom model registration documentation
checkpoints/      local model weights (not committed)
objects/          exported objects (not committed)
screenshots/      generated screenshots (not committed)
```

## Planned media

Screenshots and demonstration video will be added separately.

## Privacy / local processing

After the required packages and selected model weights are installed, normal image/video inference can run locally. Input frames do not need to be uploaded to a cloud service.

## License and upstream attribution

Original application code in this repository is released under the **Apache License 2.0**. See [LICENSE](LICENSE).

The project uses the OpenMMLab ecosystem as external dependencies:

- MMDetection 3.3.0 — Apache License 2.0
- MMCV 2.1.0 — Apache License 2.0
- MMEngine — Apache License 2.0
- RTMDet / RTMDet-Ins / Mask R-CNN checkpoints — downloaded separately and subject to their applicable upstream terms

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Responsible use

Computer-vision models can produce incorrect classes, boxes, masks, tracking IDs, motion states and color estimates. Validate results in the intended environment before using them for safety-critical, security-critical or automated decisions.
