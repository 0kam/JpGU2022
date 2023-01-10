from scripts.doublecnn import DoubleCNNClassifier
model = DoubleCNNClassifier("configs/AJG2023Spr/config_ground_2016.yaml")
model.kfold(30, 5)
model = DoubleCNNClassifier("configs/AJG2023Spr/config_ground_2020.yaml")
model.kfold(30, 5)