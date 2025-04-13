import os
import shutil

dst_path  = "/home/ypj/code/research/surgical desmoking/atomosphere cyclegan"
root_path = "/home/ypj/code/research/surgical desmoking/atomosphere cyclegan/checkpoints"
modes = ['light', 'middle', 'heavy']
models = ["cycle_gan", "atmosphere_cycle_gan", "pix2pix", "atmosphere_pix2pix"]
# models = ["pix2pix", "atmosphere_pix2pix"]
# models = ["pix2pix","cycle_gan", "atmosphere_pix2pix", "atmosphere_cycle_gan"]
netGs = ["PFAN", "swinunet", "unet_256", "unet_128", "resnet_9blocks", "resnet_6blocks"]

if not os.path.exists(os.path.join(dst_path, "logs")):
    os.mkdir(os.path.join(dst_path, "logs"))

for i in range(len(models)):
    for j in range(len(modes)):
        folder = models[i].capitalize() + modes[j].capitalize()

        if not os.path.exists(os.path.join(dst_path, "logs", folder)):
            os.mkdir(os.path.join(dst_path, "logs", folder))


for i in range(len(models)):
    for j in range(len(modes)):
        for k in range(len(netGs)):
            model = models[i]
            mode = modes[j]
            netG = netGs[k]
            name = model.capitalize() + netG.capitalize() + mode.capitalize()
            log_path = os.path.join(root_path, name, "log.txt")

            if os.path.exists(log_path):
                shutil.copy(log_path, os.path.join(dst_path, "logs", model.capitalize() + mode.capitalize(), netG.capitalize() + "_log.txt"))


