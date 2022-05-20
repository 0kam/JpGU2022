import argparse
from PIL import Image
from glob import glob
from os import path
from tqdm import tqdm

def as_gif(in_dir, out_path, size=(800, 600), duration=500, loop=0):
    in_dir = path.abspath(in_dir)
    out_path = path.abspath(out_path)
    print("input directory : {}".format(in_dir))
    print("out path : {}".format(out_path))
    if path.exists(in_dir) == False:
        raise ValueError("{} does not exists!".format(in_dir))
    print("Loading images......")
    images = [Image.open(i).convert("P", palette=Image.ADAPTIVE, colors=256) for i in tqdm(sorted(glob(in_dir + "/*")))]
    n_img = len(images)
    if n_img < 2:
        raise ValueError("{} has only {} image!".format(in_dir, n_img))
    else:
        print("Found {} images!".format(n_img))
    images = [i.resize(size, Image.BICUBIC) for i in images]
    print("Saving as gif......")
    images[0].save(out_path, save_all=True, append_images=images[1:],  duration=duration, loop=loop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating a GIF animation from images")

    # Require parameters
    parser.add_argument("--in_dir", "-i", help="The path to the directory that stores input images", type=str, required=True)
    parser.add_argument("--out_path", "-o", help="The output path. e.g., 'my_animation.gif'", type=str, required=True)
    
    # Optional parameters
    parser.add_argument("--size", "-s", help="The output size of each frame. (width, height). Default: (800, 600)", type=int, nargs=2, default=[800, 600])
    parser.add_argument("--duration", "-d", help="The display duration of each frame of the multiframe gif, in milliseconds. Default: 500", type=int, default=500)
    parser.add_argument("--loop", "-l", help="Integer number of times the GIF should loop. 0 means that it will loop forever. Default: 0", type=int, default=0)
    
    args = parser.parse_args() 
    
    as_gif(args.in_dir, args.out_path, args.size, args.duration, args.loop)