from scripts.doublecnn import DoubleCNNClassifier

# sky
## train with 2016, test on 2020 
model_sky = DoubleCNNClassifier("configs/AJG2023Spr/config_sky_2016.yaml")
model_sky.load("runs/efficientnet_b4_sky_20230105_16_39_36/best.pth")
loss_sky, res_sky = model_sky.test("data/cyo/2020/train_sky", "data/cyo/2020/labeled_CYO_2020_sky.csv", "results/sky_2016_2020.yaml")

## train with 2020, test on 2016
model_sky = DoubleCNNClassifier("configs/AJG2023Spr/config_sky_2020.yaml")
model_sky.load("runs/efficientnet_v2_s_sky_20230110_12_40_50/best.pth")
loss_sky, res_sky = model_sky.test("data/cyo/2016/train_sky", "data/cyo/2016/labeled_CYO_2016_sky.csv", "results/sky_2020_2016.yaml")

# sky
## train with 2016, test on 2020 
model_ground = DoubleCNNClassifier("configs/AJG2023Spr/config_ground_2016.yaml")
model_ground.load("runs/efficientnet_b4_ground_20230110_20_46_30/best.pth")
loss_ground, res_ground = model_ground.test("data/cyo/2020/train_ground", "data/cyo/2020/labeled_CYO_2020_ground.csv", "results/ground_2016_2020.yaml")

## train with 2020, test on 2016
model_ground.load("runs/efficientnet_v2_s_ground_20230110_16_10_39/best.pth")
model_ground = DoubleCNNClassifier("configs/AJG2023Spr/config_ground_2016.yaml")
loss_ground, res_ground = model_ground.test("data/cyo/2016/train_ground", "data/cyo/2016/labeled_CYO_2016_ground.csv", "results/ground_2020_2016.yaml")