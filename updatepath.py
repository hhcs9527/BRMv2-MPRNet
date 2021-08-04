import os
from PIL import Image

def change_path(experiment):
    path = f'./checkpoints/Deblurring/results/{experiment}/GoPro/'
    for i, p in enumerate(sorted(os.listdir(path))):
        img = Image.open(f'{path}{p}')
        img.save(f'/home/hh/Desktop/disk2/MPGAN/runs/{experiment}/result_picture/deblur/{i}.png')        
        #os.rename(os.path.join(path, p), os.path.join(path, f'{i}.png'))

if __name__ == '__main__':
    experiment = 'MPRNet'
    change_path(experiment)