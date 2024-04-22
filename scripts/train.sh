accelerate launch --config_file config/train.yaml train.py \
--train_meta_file dataset/FIGR-SVG-train.csv \
--val_meta_file dataset/FIGR-SVG-valid.csv \
--svg_folder dataset/FIGR-SVG-svgo \
--output_dir proj_log/ \
--project_name FIGR_SVG512_TEXT50_D16_H512_MLP1024 \
--maxlen 512 \
--batchsize 20