from tqdm import tqdm
from glob import glob
from scripts.utils import filter_images, sample_images, draw_box_ridge

#filter_images("data/cyo/2019", 7, 16)
#sample_images("data/cyo/2019", "data/cyo/2019_sampled", 200)

for path in tqdm(sorted(glob("data/cyo/2016/sampled/*"))):
    draw_box_ridge(path, ridge_upper=775, ridge_lower=935, num_split_sky=(2,2), num_split_ground=(4,4))


for path in tqdm(sorted(glob("data/cyo/2020/sampled/*"))):
    draw_box_ridge(path, ridge_upper=777, ridge_lower=935, num_split_sky=(2,2), num_split_ground=(4,4))