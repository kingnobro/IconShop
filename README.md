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

Download the labels and SVG files from [Google Drive](https://drive.google.com/drive/folders/1dXdrBqJDmEn8K8TeY2w3mvEtADZipPOc?usp=drive_link) and place them in the `./dataset` folder. You need to unzip the `FIGR-SVG-svgo.zip`.

The resulting file structure should be:
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

There are some other useful files in the Google Drive link. Please check the `Readme.md` and download them as needed.

## Training
I use 8 RTX 3090 (or 2 A100) to train the model.

Feel free to adjust the batch size, number of epochs, and learning rate. The training will take several days, so I did not spend a lot of time optimizing these parameters.
```
bash scripts/train.sh
```

Use the following command to check the loss curve:
```
bash scripts/log.sh
```

## Sample
```bash
bash scripts/sample.sh
```
Download our [pretrained models](https://drive.google.com/drive/folders/1xF0AjYk-WvfNuv6z5xluNDC87ktke2rK?usp=sharing) and unzip it under `proj_log/FIGR_SVG`. This model is trained with 100 epochs.

Please note that the results presented in the paper may differ from those of this pretrained model (e.g., the same prompt will create different outputs). Due to copyright issues, access to the model described in the paper is available upon request for research purposes only.

## Miscellaneous
- Our code is based on [SkexGen](https://github.com/samxuxiang/SkexGen) and [DeepSVG](https://github.com/alexandre01/deepsvg) (for read and write SVG data).
- Acknowledgement: We would like to express our sincere gratitude to OPPO for their generous support of our work.