from scripts.doublecnn import DoubleCNNClassifier
d = "/media/okamoto/HDD4TB/JpGU2022/"

# sky
## train with 2016, test on 2020 
model_sky = DoubleCNNClassifier("configs/AJG2023Spr/config_sky_2016.yaml")
model_sky.load("runs/efficientnet_b4_sky_20230111_05_06_08/best.pth")
loss_sky, res_sky = model_sky.test(d + "data/cyo/2020/train_sky", d + "data/cyo/2020/labeled_CYO_2020_sky.csv", "results/sky_2016_2020.yaml")

## train with 2020, test on 2016
model_sky = DoubleCNNClassifier("configs/AJG2023Spr/config_sky_2020.yaml")
model_sky.load("runs/efficientnet_b4_sky_20230111_06_10_56/best.pth")
loss_sky, res_sky = model_sky.test(d + "data/cyo/2016/train_sky", d + "data/cyo/2016/labeled_CYO_2016_sky.csv", "results/sky_2020_2016.yaml")

# ground
## train with 2016, test on 2020 
model_ground = DoubleCNNClassifier("configs/AJG2023Spr/config_ground_2016.yaml")
model_ground.load("runs/efficientnet_b4_ground_20230110_20_58_35/best.pth")
loss_ground, res_ground = model_ground.test(d + "data/cyo/2020/train_ground", d + "data/cyo/2020/labeled_CYO_2020_ground.csv", "results/ground_2016_2020.yaml")
df = model_ground.predict_dir(d + "data/cyo/2020/sampled", "sampled", "predicted_ground_sampled", use_best=False)
df = model_ground.predict_dir(d + "data/cyo/2016/2016", "2016/2016", "2016/predicted_ground", use_best=False)
df.to_csv(d + "data/cyo/2016/predicted_ground.csv")

## train with 2020, test on 2016
model_ground = DoubleCNNClassifier("configs/AJG2023Spr/config_ground_2020.yaml")
model_ground.load("runs/efficientnet_b4_ground_20230111_01_01_21/best.pth")
loss_ground, res_ground = model_ground.test(d + "data/cyo/2016/train_ground", d + "data/cyo/2016/labeled_CYO_2016_ground.csv", "results/ground_2020_2016.yaml")
df = model_ground.predict_dir(d + "data/cyo/2016/sampled", "sampled", "predicted_ground", use_best=False)
df = model_ground.predict_dir(d + "data/cyo/2020/2020", "2020/2020", "2020/predicted_ground", use_best=False)
df.to_csv(d + "data/cyo/2020/predicted_ground.csv")