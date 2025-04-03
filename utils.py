import time

import torch
from PIL import Image
from thop import profile
import os
import numpy as np
from matplotlib import pyplot as plt


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
        lr = param_group['lr']
    return lr


def adjust_lr_2(optimizer, epoch_list, lr_list, epoch):
    if epoch < epoch_list[0]:
        c_lr = lr_list[0]
    elif epoch < sum(epoch_list[:-1]):
        c_lr = lr_list[1]
    else:
        c_lr = lr_list[2]
    for param_group in optimizer.param_groups:
        param_group['lr'] = c_lr
        lr = param_group['lr']
    return lr


def opt_save(opt):
    log_path = opt.save_path + "train_settings.log"
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    att = [i for i in opt.__dir__() if not i.startswith("_")]
    with open(log_path, "w") as f:
        for i in att:
            print("{}:{}".format(i, eval(f"opt.{i}")), file=f)


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


class Timer(object):
    def __init__(self):
        self.time = 0

    def record(self):
        self.time = time.time() * 1000

    def elapsed_time(self, ender):
        return ender.time - self.time


def fps(model, epoch_num, size, gpu=0, param_count=2):
    assert param_count in [1, 2]

    ls = []  # 每次计算得到的fps
    iterations = 300  # 重复计算的轮次
    device = f'cuda:{gpu}' if gpu >= 0 else 'cpu'
    model.eval().to(device)
    random_input = torch.randn(1, 3, size, size).to(device)
    starter, ender = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if gpu >= 0 else (
        Timer(), Timer())

    input_tensor = [random_input] if param_count == 1 else [random_input, random_input]

    for i in range(epoch_num + 1):
        # 测速
        times = torch.zeros(iterations)  # 存储每轮iteration的时间
        with torch.inference_mode():
            for iter in range(iterations):
                starter.record()
                _ = model(*input_tensor)
                ender.record()

                # 同步GPU时间
                torch.cuda.synchronize() if gpu >= 0 else None
                curr_time = starter.elapsed_time(ender)  # 计算时间
                times[iter] = curr_time
                # print(curr_time)

        mean_time = times.mean().item()

        if i == 0:
            print("Initialization Inference time: {:.2f} ms, FPS: {:.2f} ".format(mean_time, 1000 / mean_time))
        if i != 0:
            ls.append(1000 / mean_time)
            print("{}/{} Inference time: {:.2f} ms, FPS: {:.2f} ".format(i, epoch_num, mean_time, 1000 / mean_time))
    print(f"平均fps为 {np.mean(ls):.2f}")
    print(f"最大fps为 {np.max(ls):.2f}")


def flops(model, size, gpu=0, param_count=2):
    assert param_count in [1, 2], 'please input correct param number !'
    model.cuda().eval()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    dummy_input = torch.randn(1, 3, size, size).cuda()
    if param_count == 2:
        flops, params = profile(model, (dummy_input, dummy_input))
    else:
        flops, params = profile(model, (dummy_input,))
    print('GFLOPs: %.3f , params: %.2f M' % (flops / 1.0e9, params / 1.0e6))


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def param(model):
    param_sum = 0
    for i in model.named_modules():
        if '.' not in i[0] and i[0] != '':
            layer = getattr(model, i[0])
            temp = sum(p.numel() for p in layer.parameters() if p.requires_grad) / 1e6
            param_sum += temp
            print(i[0], temp, "M")
    print(param_sum, 'M')


def load_param(pth_path, model, mode):
    if mode == "rgb":
        pretraind_dict = torch.load(pth_path)["state_dict"]
    else:
        pretraind_dict = torch.load(pth_path)
    model_dict = model.state_dict()

    # 只将pretraind_dict中那些在model_dict中的参数，提取出来
    state_dict = {k: v for k, v in pretraind_dict.items() if k in model_dict.keys()}
    # 将提取出来的参数更新到model_dict中，而model_dict有的，而state_dict没有的参数，不会被更新
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    len_pretrained = len(pretraind_dict.keys())
    len_model = len(model_dict.keys())
    len_match = len(state_dict.keys())

    print(f"{mode=} {len_match} / {len_model} / {len_pretrained}")





def resize_dictionary(path):
    method_images = sorted(os.listdir(path))
    # print(method, len(method_images))

    for image in method_images:
        img_path = os.path.join(path, image)
        img = Image.open(img_path)
        img = img.resize((256, 256))
        img.save(img_path)


def visualize_feature_map(feature_map, folder_name, output_path="/DataSet/SOD/RGB-D SOD/test_data/vis", resize=True,
                          batch_index=0):
    """
    可视化并保存特征图。

    参数：
    - feature_map: 形状为 B,C,H,W 的 PyTorch 张量。
    - folder_name: 保存特征图的子文件夹名称。
    - output_path: 保存特征图的根目录。
    - resize: 是否重设可视化后的特征图大小，默认为256*256。
    - batch_index: 要可视化的批次索引，默认为 0。

    返回：
    - 无返回值，直接将特征图保存为图像文件。
    """
    # 检查输入张量的形状是否为四维
    if len(feature_map.shape) != 4:
        raise ValueError("输入的 feature_map 必须是 4D 张量 (B, C, H, W)。")

    # 选择指定批次索引的特征图
    feature = feature_map[batch_index]  # 形状为 C, H, W

    # 将张量移动到 CPU 并转换为 numpy 数组
    feature = feature.cpu().detach().numpy()

    # 获取通道数
    num_channels = feature.shape[0]

    # 创建保存路径
    save_path = os.path.join(output_path, folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 遍历每个通道并保存特征图
    for i in range(num_channels):
        plt.figure()
        plt.imshow(feature[i], cmap='jet')
        plt.axis('off')
        # 保存图像
        img_path = os.path.join(save_path, f"channel_{i + 1}.png")
        plt.imsave(img_path, feature[i], cmap='jet')
        plt.close()

    resize_dictionary(save_path) if resize else None
    print(f"\n{folder_name} visualization finished!")


if __name__ == '__main__':
    from model.LESOD import LESOD

    ################## LESOD # ##################
    TRAIN_SIZE = 384

    model = LESOD().eval()
    x = torch.randn(1, 3, TRAIN_SIZE, TRAIN_SIZE)
    f_ls = model(x, x)

    for i in f_ls:
        print(i.shape)

    fps(model=model, epoch_num=5, size=TRAIN_SIZE, gpu=0, param_count=2)

    param(model)
    flops(model, TRAIN_SIZE, gpu=0, param_count=2)
