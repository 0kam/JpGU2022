from scripts.doublecnn import DoubleCNNClassifier

model_sky = DoubleCNNClassifier("/home/okamoto/JpGU2022/scripts/config_sky.yaml")
model_sky.load("/home/okamoto/JpGU2022/runs/efficientnet_b4_20220517_12_32_28/best.pth")
df = model_sky.predict_dir("/home/okamoto/JpGU2022/data/cyo/2019/", "2019", "2019_predicted_sky", use_best=False)
df.to_csv("/home/okamoto/JpGU2022/data/cyo/2019_sky.csv")


model_ground = DoubleCNNClassifier("/home/okamoto/JpGU2022/scripts/config_ground.yaml")
model_ground.load("/home/okamoto/JpGU2022/runs/efficientnet_b4_20220516_18_00_21/best.pth")
df = model_ground.predict_dir("/home/okamoto/JpGU2022/data/cyo/2019/", "2019", "2019_predicted_ground", use_best=False)
df.to_csv("/home/okamoto/JpGU2022/data/cyo/2019_ground.csv")