from scripts.doublecnn import DoubleCNNClassifier

model_ground = DoubleCNNClassifier("/home/okamoto/JpGU2022/scripts/config_sky.yaml")
model_ground.kfold(200, 5)