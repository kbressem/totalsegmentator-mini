Before you train, it is advised to remove previously cached files. In some cases, if the training crashes, corrupted files might be created, leading to cryptic error messages in subsequent trainings. 

## Inference

```bash
python -m monai.bundle run evaluating \
  --meta_file configs/metadata.yaml \
  --config_file configs/inference.yaml \
  --logging_file configs/logging.conf
```
To run inference on a single file or a dicretory containing multiple image files use the `--datadir` flag

## Evaluation

```bash
python -m monai.bundle run evaluating \
  --meta_file configs/metadata.yaml \
  --config_file configs/evaluate.yaml \
  --logging_file configs/logging.conf
```
To run inference on a single file or a dicretory containing multiple image files use the `--datadir` flag

## Training

During training, this bundle saves both, the model weights AND the optmizer in `model.pt`. This can be an isse, e.g. if deployed in MONAI Label. Use `scripts/separate_model_optim.py` to separate them. 

### Single GPU training

```bash
python -m monai.bundle run training \
  --meta_file configs/metadata.yaml \
  --config_file "['configs/train.yaml','configs/unet.yaml']" \
  --logging_file configs/logging.conf
```

### Multi GPU trainig

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run training \
  --meta_file configs/metadata.yaml \
  --config_file "['configs/train.yaml','configs/unet.yaml','configs/multi_gpu_train.yaml']" \
  --logging_file configs/logging.conf
```
