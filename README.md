# Distributed Computing, Parallel Processing, and Object Detection

**Note:** All work in this repository is authored by Darien Nouri.

This repository contains a collection of Jupyter notebooks for various topics in distributed computing, parallel processing, and object detection.

## File Structure

```text
├── 01_PyTorch_DataParallelism.ipynb
├── 02_Staleness_ParameterServer_AsyncSGD.ipynb
├── 03_SSD_ONNX_Object_Detection.ipynb
```

## Notebooks

- **01_PyTorch_DataParallelism.ipynb**
  - Experiments with PyTorch's DataParallel Module for Synchronous SGD across multiple GPUs.
  - Analyzes training time, scalability, and communication bandwidth utilization.

- **02_Staleness_ParameterServer_AsyncSGD.ipynb**
  - Calculates staleness in a Parameter-Server based Asynchronous SGD training system with two learners.
  - Examines the number of weight updates between reading and updating weights for each gradient calculation.

- **03_SSD_ONNX_Object_Detection.ipynb**
  - Explores inferencing using the SSD ONNX model with the ONNX Runtime Server.
  - Includes model testing, fine-tuning, conversion to ONNX, and running inferencing using ONNX Runtime.