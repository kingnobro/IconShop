# IconShop: Text-Guided Vector Icon Synthesis with Autoregressive Transformers

# Setup
```
git clone https://github.com/kingnobro/IconShop.git
cd IconShop
```

## Environment
To set up our environment, please run:
```
conda env create -f environment.yml
conda activate iconshop
```
Please refer to [cairosvg](https://cairosvg.org/documentation/#installation)'s documentation for additional requirements of CairoSVG. For example:
- on Ubuntu: `sudo apt-get install libcairo2-dev`.
- on macOS: `brew install cairo libffi`.

## Dataset
We have cleaned up the [FIGR-8-SVG](https://github.com/marcdemers/FIGR-8-SVG) by command simplification, removing the black bounding box, and resizing the icons.

Download the labels and SVG files from [Google Drive](https://drive.google.com/drive/folders/1dXdrBqJDmEn8K8TeY2w3mvEtADZipPOc?usp=drive_link) and place them in the `./dataset` folder. The resulting file structure should be:
```
./dataset
├── FIGR-SVG-train.csv
├── FIGR-SVG-valid.csv
├── FIGR-SVG-test.csv
└── FIGR-SVG-svgo
    ├── 100000-200.svg
    ├── 1000003-200.svg
    └── ...
```

> We offer additional useful files at these links. Please download them as needed.

## Training
I use 8 RTX 3090 (or 2 A100) to train the model.
```
bash scripts/train.sh
```

Moniter the loss:
```
bash scripts/log.sh
```

> Feel free to adjust the batch size, number of epochs, and learning rate, as I cannot optimize these parameters effectively with limited GPU resources.

## Sample
```bash
bash scripts/sample.sh
```
