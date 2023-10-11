# SatViT
Vision Transformers for Crop Type Recognition

## Container
```
git clone https://github.com/zlab-foss/satvit.git  
cd satvit
docker run -it --name 'satvit' -v /media/data/active/PASTIS/PASTIS9/:/workspace/PASTIS9 -v /media/data/active/PASTIS/PASTIS24/:/workspace/PASTIS24 -v $(pwd):/workspace/satvit --gpus all satvit:0.0.1
```

## Data
```
docker run -it --rm -v $(pwd):/workspace pastis:pytorch-2.0.0-cuda11.7-cudnn8-runtime
python data2windows.py --rootdir PASTIS --savedir $(pwd)/outputDirectory --HWout 9
```

## Training
`python train.py --config_file ./configs/classification.yaml`

## Evaluation