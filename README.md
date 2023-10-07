# SatViT
Vision Transformers for Crop Type Recognition


## Data
`python data2windows.py --rootdir PASTIS --savedir $(pwd) --HWout 9`

## Training
`python train.py --config_file ./configs/classification.yaml`

## Evaluation