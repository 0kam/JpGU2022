from scripts.doublecnn import DoubleCNNClassifier
import tensorboardX

model_sky = DoubleCNNClassifier("/home/okamoto/JpGU2022/scripts/config_sky.yaml")
model_sky.train(30)

model_sky.load("/home/okamoto/JpGU2022/runs/efficientnet_b4_20220302_12_29_44/best.pth")
df = model_sky.predict_dir("/home/okamoto/JpGU2022/data/cyo/2019/", "2019", "2019_predicted_sky", use_best=True)
df.to_csv("/home/okamoto/JpGU2022/data/cyo/2019_test_sky.csv")

model_ground = DoubleCNNClassifier("/home/okamoto/JpGU2022/scripts/config_ground.yaml")
model_ground.train(30)
model_ground.kfold(200, 5)

model_ground.load("/home/okamoto/JpGU2022/runs/efficientnet_b4_20220302_12_59_08/best.pth")
df = model_ground.predict_dir("/home/okamoto/JpGU2022/data/cyo/2019/", "2019", "2019_predicted_ground", use_best=True)
df.to_csv("/home/okamoto/JpGU2022/data/cyo/2019_test_ground.csv")