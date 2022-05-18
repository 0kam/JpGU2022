from tqdm import tqdm
from glob import glob
from scripts.utils import filter_images, sample_images, draw_box_ridge

#filter_images("data/cyo/2019", 7, 16)
#sample_images("data/cyo/2019", "data/cyo/2019_sampled", 200)

for path in tqdm(sorted(glob("data/cyo/2019_sampled/*"))):
    draw_box_ridge(path, ridge_upper=786, ridge_lower=944, num_split_sky=(4,2), num_split_ground=(4, 3))