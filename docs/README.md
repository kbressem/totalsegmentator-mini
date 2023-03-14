# Totalsegmentator Mini

Totalsegmentator Mini is a small-scale clone of [Totalsegmentator](https://github.com/wasserth/TotalSegmentator), a tool that utilizes the MONAI deep learning framework to perform semantic segmentation on medical images. This tool can be used for both inference and training tasks.

<img src="https://user-images.githubusercontent.com/37253540/216309343-ab6e3d64-2f13-43b4-93c0-4fa85e8e57fa.png"  width="300" height="300">

This model was trained on the TotalSegmentator dataset as well as on 3500 additional abdominal CT scans, which were first segmented by the full TotalSegmentator model. 

## Overview of bundle commands
### Inference

```bash
python -m monai.bundle run inference \
  --meta_file configs/metadata.yaml \
  --config_file configs/inference.yaml \
  --logging_file configs/logging.conf \
  --dataset_dir <path_to_file_or_folder>
```
To run inference on a single file or a directory containing multiple image files use the `--dataset_dir` flag. 
Nested directories are not supported.   
To control where the files are stored, overwrite the output directory with the `--output_dir` flag. 

### Evaluation
Evaluate the model on the official total segmentator training data (you need to download the data and adapt the paths beforehand). 
```bash
python -m monai.bundle run evaluating \
  --meta_file configs/metadata.yaml \
  --config_file configs/evaluate.yaml \
  --logging_file configs/logging.conf
```

### Training

During training, this bundle saves both, the model weights AND the optimizer in `model.pt`. This can be an issue, e.g. if deployed in MONAI Label. Use `scripts/separate_model_optim.py` to separate them. 

#### Single GPU training

```bash
python -m monai.bundle run training \
  --meta_file configs/metadata.yaml \
  --config_file "['configs/train.yaml','configs/unet.yaml']" \
  --logging_file configs/logging.conf
```

#### Multi GPU trainig

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run training \
  --meta_file configs/metadata.yaml \
  --config_file "['configs/train.yaml','configs/unet.yaml','configs/multi_gpu_train.yaml']" \
  --logging_file configs/logging.conf
```
By using Totalsegmentator Mini, you can perform semantic segmentation on medical images in a quick and efficient manner.
