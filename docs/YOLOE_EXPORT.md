# YOLOE to Objects365 Conversion Utility

This utility converts YOLOE (YOLO Enterprise) segmentation models to standard detection models predicting only Objects365 classes.

## Background

Based on GitHub issues:
- **#22630**: Request for YOLO11 pretrained models on Objects365
- **#22744**: YOLOE seg→det conversion pattern

## Usage

### Convert YOLOE Segmentation to Objects365 Detection

```bash
# Convert YOLOE segmentation model to Objects365 detection model
python -m src.yoloe_utils convert yoloe-11s-seg.pt

# With custom output path
python -m src.yoloe_utils convert yoloe-11s-seg.pt -o models/yoloe-11s-objects365.pt

# Using specific device
python -m src.yoloe_utils convert yoloe-11l-seg.pt --device cuda
```

### Export to Deployment Formats

```bash
# Export to ONNX (default)
python -m src.yoloe_utils export yoloe-11s-objects365.pt

# Export to TensorRT with FP16
python -m src.yoloe_utils export yoloe-11s-objects365.pt \
    --format engine --half --device cuda

# Export to CoreML for iOS
python -m src.yoloe_utils export yoloe-11s-objects365.pt --format coreml

# Export to TensorFlow Lite
python -m src.yoloe_utils export yoloe-11s-objects365.pt --format tflite
```

## Python API

```python
from src.yoloe_utils import convert_yoloe_to_objects365, export_model

# Convert model
model_path = convert_yoloe_to_objects365(
    "yoloe-11s-seg.pt",
    output_path="yoloe-11s-objects365.pt"
)

# Export to ONNX
export_model(
    "yoloe-11s-objects365.pt",
    format="onnx",
    imgsz=640,
    half=False,
    simplify=True
)
```

## Objects365 Classes

The converted model predicts all 365 Objects365 classes including:
- **Bottle** (Class 8)
- **Canned** (Class 64) - for cans
- **Cup** (Class 10)
- **Wine Glass** (Class 35)
- And 361 other everyday objects

See `src/yoloe_utils.py` for the full class list.

## Model Export Parameters Explained

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | str | "onnx" | Export format. See formats table below. |
| `imgsz` | int | 640 | Input image size (height and width). Must match training size for best results. |
| `device` | str | "cpu" | Device for export: "cpu", "cuda", "mps" (Apple Silicon). |
| `output` | str | None | Custom output path. If None, uses default naming. |

### Precision & Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `half` | bool | False | **FP16 (Half Precision)**. Export model in 16-bit floating point. Reduces model size by 50% and can speed up inference on GPUs with Tensor Cores. Requires GPU for export. |
| `simplify` | bool | True | **ONNX Simplification**. Removes redundant operations from ONNX graph. Recommended for deployment as it improves compatibility and may speed up inference. |
| `opset` | int | 12 | **ONNX Opset Version**. Version of ONNX operator set (9-17). Higher versions support more operators. Use 12 for maximum compatibility, 17 for latest features. |

### Shape & Batching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dynamic` | bool | False | **Dynamic Input Shapes**. When True, allows variable batch sizes and image dimensions at inference time. Required for processing images of different sizes. When False, model is optimized for fixed `imgsz`. |
| `nms` | bool | False | **Embed NMS**. Embeds Non-Maximum Suppression directly in the exported model. Useful for end-to-end deployment but limits post-processing flexibility. |

### Export Formats

| Format | Value | Extension | Best For |
|--------|-------|-----------|----------|
| ONNX | `"onnx"` | `.onnx` | Universal deployment, cross-platform |
| TorchScript | `"torchscript"`, `"ts"` | `.torchscript` | PyTorch mobile, C++ inference |
| TensorRT | `"engine"`, `"trt"` | `.engine` | NVIDIA GPU acceleration |
| OpenVINO | `"openvino"`, `"xml"` | `.xml` + `.bin` | Intel CPUs and VPUs |
| CoreML | `"coreml"` | `.mlpackage` | Apple devices (iOS, macOS) |
| TensorFlow SavedModel | `"saved_model"`, `"pb"` | SavedModel dir | TensorFlow ecosystem |
| TensorFlow Lite | `"tflite"` | `.tflite` | Mobile and embedded devices |
| Edge TPU | `"edgetpu"` | `_edgetpu.tflite` | Google Coral devices |
| PaddlePaddle | `"paddle"` | Paddle format | Baidu Paddle ecosystem |
| NCNN | `"ncnn"` | NCNN format | Mobile/embedded (Tencent) |

## Detailed Parameter Explanations

### `half` (FP16 Precision)

```python
# FP32 (default) - full precision
model.export(format="onnx", half=False)  # ~100MB

# FP16 - half precision
model.export(format="onnx", half=True)   # ~50MB, faster on modern GPUs
```

**When to use:**
- ✅ NVIDIA GPUs with Tensor Cores (RTX 20 series, A100, etc.)
- ✅ Edge devices with FP16 support
- ❌ Older GPUs without FP16 support
- ❌ When maximum precision is required

### `simplify` (ONNX Only)

```python
# With simplification (recommended)
model.export(format="onnx", simplify=True)

# Without simplification
model.export(format="onnx", simplify=False)
```

**What it does:**
- Removes redundant `Identity` operations
- Fuses consecutive operations (Conv+BN+ReLU)
- Constant folds constant expressions
- Improves inference speed and compatibility

### `opset` (ONNX Only)

```python
# Conservative - maximum compatibility
model.export(format="onnx", opset=12)  # Works with older runtimes

# Modern - latest features
model.export(format="onnx", opset=17)  # Newer operators, may not work everywhere
```

**Common opset versions:**
- **9**: ONNX 1.4 (minimum recommended)
- **11**: ONNX 1.6 (good baseline)
- **12**: ONNX 1.7 (recommended for compatibility)
- **17**: ONNX 1.13 (latest, required for some operations)

### `dynamic` (ONNX/TorchScript)

```python
# Static shapes - optimized for fixed size
model.export(format="onnx", dynamic=False)  # Input must be 640x640

# Dynamic shapes - flexible size
model.export(format="onnx", dynamic=True)   # Can handle any size
```

**Trade-offs:**
- **Static (False)**: Better optimization, faster inference, fixed size
- **Dynamic (True)**: Flexible input sizes, slightly slower, more compatible

### `nms` (End-to-End)

```python
# Standard export - NMS done in post-processing
model.export(format="onnx", nms=False)

# End-to-end - NMS embedded in model
model.export(format="onnx", nms=True)
```

**When to embed NMS:**
- ✅ Simple deployment pipelines
- ✅ When you want raw output to be final detections
- ❌ When you need custom NMS parameters
- ❌ When doing additional post-processing

## Recommended Export Configurations

### For Web (ONNX Runtime Web)

```python
export_model(
    "model.pt",
    format="onnx",
    imgsz=640,
    half=False,      # WebGPU may not support FP16
    simplify=True,
    opset=12,        # Conservative for browser compatibility
    dynamic=False,   # Fixed size for optimization
)
```

### For NVIDIA GPU (TensorRT)

```python
export_model(
    "model.pt",
    format="engine",
    imgsz=640,
    half=True,       # FP16 for Tensor Cores
    device="cuda",
    dynamic=False,   # Static for maximum speed
)
```

### For iOS (CoreML)

```python
export_model(
    "model.pt",
    format="coreml",
    imgsz=640,
    half=True,       # FP16 supported on Apple Neural Engine
    nms=True,        # End-to-end for simpler pipeline
)
```

### For Edge/Mobile (TFLite)

```python
export_model(
    "model.pt",
    format="tflite",
    imgsz=320,       # Smaller for mobile
    half=False,      # INT8 quantization preferred
)
```

## Troubleshooting

### NaN outputs in FP16

Issue #22744 reported NaN outputs when exporting YOLOE to FP16 ONNX.

**Workarounds:**
1. Use FP32: `half=False`
2. Try different opset: `opset=12` vs `opset=17`
3. Disable simplification: `simplify=False`

### Large model size

```python
# Use FP16
export_model("model.pt", format="onnx", half=True)

# For TFLite, use INT8 quantization
export_model("model.pt", format="tflite", int8=True)
```

### Slow inference

```python
# Use TensorRT for NVIDIA GPUs
export_model("model.pt", format="engine", half=True)

# Use OpenVINO for Intel CPUs
export_model("model.pt", format="openvino")
```

## References

- [Ultralytics Export Documentation](https://docs.ultralytics.com/modes/export/)
- [ONNX Documentation](https://onnx.ai/)
- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [CoreML Tools](https://apple.github.io/coremltools/)
