from scripts.doublecnn import DoubleCNNClassifier
model = DoubleCNNClassifier("configs/AJG2023Spr/config_sky_2016.yaml")
model.kfold(30, 5)
model = DoubleCNNClassifier("configs/AJG2023Spr/config_sky_2020.yaml")
model.kfold(30, 5)