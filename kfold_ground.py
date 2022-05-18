from scripts.doublecnn import DoubleCNNClassifier
model_ground = DoubleCNNClassifier("/home/okamoto/JpGU2022/scripts/config_ground.yaml")
model_ground.kfold(200, 5)