from scripts.doublecnn import DoubleCNNClassifier

model_sky = DoubleCNNClassifier("/home/okamoto/JpGU2022/scripts/config_sky.yaml")
model_sky.load("/home/okamoto/JpGU2022/runs/efficientnet_b4_20220517_12_32_28/best.pth")
df = model_sky.predict_dir("/home/okamoto/JpGU2022/data/cyo/2019/", "2019", "2019_predicted_sky", use_best=False)
df.to_csv("/home/okamoto/JpGU2022/data/cyo/2019_sky.csv")

# ground
## train with 2016, test on 2020 
model_ground = DoubleCNNClassifier("configs/AJG2023Spr/config_ground_2016.yaml")
model_ground.load("runs/efficientnet_b4_ground_20230110_20_58_35/best.pth")
loss_ground, res_ground = model_ground.test("data/cyo/2020/train_ground", "data/cyo/2020/labeled_CYO_2020_ground.csv", "results/ground_2016_2020.yaml")

## train with 2020, test on 2016
model_ground = DoubleCNNClassifier("configs/AJG2023Spr/config_ground_2016.yaml")
model_ground.load("runs/efficientnet_b4_ground_20230111_01_01_21/best.pth")
loss_ground, res_ground = model_ground.test("data/cyo/2016/train_ground", "data/cyo/2016/labeled_CYO_2016_ground.csv", "results/ground_2020_2016.yaml")