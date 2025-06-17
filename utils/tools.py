import torch
import random
import numpy as np
import os

def denormalize(image_tensor, mean, std):
    mean = torch.tensor(mean, dtype=image_tensor.dtype).to(image_tensor.device)
    std = torch.tensor(std, dtype=image_tensor.dtype).to(image_tensor.device)
    mean = mean[None, :, None, None] 
    std = std[None, :, None, None]   

    return image_tensor * std + mean


def set_seed(seed=3407):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def init_env(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

def get_tensor_distributiuon(output):
    print('output max value: %.2f, min value: %.2f' % (torch.max(output), torch.min(output)))
    print('output mean value: %.2f, median value: %.2f' % (torch.mean(output), torch.median(output)))


def show_tensor_iamge(tensor_input):
    import matplotlib.pyplot as plt

    image_index = 0

    try:
        img = tensor_input[image_index].permute(1, 2, 0).numpy()
    except:
        img = tensor_input[image_index].permute(1, 2, 0).detach().cpu().numpy()

    plt.imshow(img)
    plt.axis('off')  
    plt.show()

def show_model_predict_result(output,image_index=0,channel_index=0,output_file_path = 'output_image_values.txt'):
    import torch
    import numpy as np

    image_index = image_index
    channel_index = channel_index

    try:
        output_image = output[image_index, channel_index].numpy()
    except:
        output_image = output[image_index, channel_index].detach().cpu().numpy()

    output_image_rounded = np.round(output_image, 3)

    output_file_path =  os.path.join('result',output_file_path )
    with open(output_file_path, 'w') as f:
        for row in output_image_rounded:
            formatted_row = ' '.join(f'{value:.3f}' for value in row)
            f.write(formatted_row + '\n')
