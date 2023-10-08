# SatViT
Vision Transformers for Crop Type Recognition

## Container
```
git clone https://github.com/zlab-foss/satvit.git  
cd satvit
docker run -it --name 'satvit' -v /media/data/active/PASTIS/PASTIS9/:/workspace/PASTIS9 -v $(pwd):/workspace/satvit --gpus all deepsat:0.0.1
```

## Data
`python data2windows.py --rootdir PASTIS --savedir $(pwd) --HWout 9`

## Training
`python train.py --config_file ./configs/classification.yaml`

## Evaluation