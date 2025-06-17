import os
import random

def split_dataset(image_folder, trainval_file, test_file, split_ratio=0.8):
    all_images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    random.shuffle(all_images)
    split_index = int(len(all_images) * split_ratio)

    trainval_images = all_images[:split_index]
    test_images = all_images[split_index:]

    with open(trainval_file, 'w') as f:
        for image in trainval_images:
            f.write(os.path.splitext(image)[0] + '\n')  

    with open(test_file, 'w') as f:
        for image in test_images:
            f.write(os.path.splitext(image)[0] + '\n')  

