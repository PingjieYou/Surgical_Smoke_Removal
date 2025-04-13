import os

def generate_sh_file(output_file, dataroot, name, model, netG, num_commands=1):
    """
    生成包含 Python 指令的 .sh 文件

    参数:
        output_file (str): 输出的 .sh 文件路径
        dataroot (str): --dataroot 参数的值
        name (str): --name 参数的值
        model (str): --model 参数的值
        netG (str): --netG 参数的值
        num_commands (int): 生成的指令数量
    """
    with open(output_file, 'a') as f:
            # command = (
            #     f"python train.py --dataroot {dataroot} --name {name} "
            #     f"--model {model} --netG {netG} --netD basic --direction AtoB "
            #     f"--dataset_mode aligned --norm batch --token_projection conv "
            #     f"--embed_dim 64 --ndf 64 --ngf 64\n"
            # )
            # f.write(command)
            command = (
                f"python test.py --dataroot {dataroot} --name {name} "
                f"--model {model} --netG {netG} --netD basic --direction AtoB "
                f"--dataset_mode aligned --norm batch --token_projection conv "
                f"--embed_dim 64 --ndf 64 --ngf 64 --eval\n"
            )
            f.write(command)

# 示例用法
if __name__ == "__main__":
    # 输出文件路径

    output_file = "test_psv2rs_cyclegan.sh"
    # output_file = "test_psv2rs_pix2pix.sh"

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n")

    # 需要修改的参数
    dataroots = "/home/ypj/dataset/cholec80_desmoking/psv2rs/"
    modes = ['light', 'middle', 'heavy']
    models = ["cycle_gan", "atmosphere_cycle_gan"]
    # models = ["pix2pix", "atmosphere_pix2pix"]
    # models = ["pix2pix","cycle_gan", "atmosphere_pix2pix", "atmosphere_cycle_gan"]
    netGs = ["PFAN", "swinunet","unet_256", "unet_128", "resnet_9blocks", "resnet_6blocks"]
    # netGs = ["swinunet"]
    for mode in modes:
        for model in models:
            for net in netGs:
                dataroot = os.path.join(dataroots, mode)
                name = model.capitalize()+net.capitalize() + mode.capitalize()
                generate_sh_file(output_file, dataroot, name, model, net)