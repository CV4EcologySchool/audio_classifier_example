# Bird Vocalization Classification

Example repository for the CV4Ecology 2022 Summer School to train and test deep
learning models on audio data, using the [SSW60 Dataset](https://github.com/visipedia/ssw60).

## Installation instructions

1. Install [Conda](http://conda.io/)

2. Create environment and install requirements

```bash
conda create -n cv4ecology python=3.8 -y
conda activate cv4ecology
pip install -r requirements.txt
```

3. Download dataset

```bash
sh scripts/download_dataset.sh 
```

This downloads the [SSW60 Dataset](https://github.com/visipedia/ssw60) to the `datasets/ssw60` folder.


## Reproduce results

1. Train

```bash
python audio_classifier/train.py --config configs/exp_resnet18.yaml
```

2. Test/inference

@CV4Ecology participants: Up to you to figure that one out. :)
