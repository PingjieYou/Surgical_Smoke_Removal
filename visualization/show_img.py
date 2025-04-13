import matplotlib.pyplot as plt

img_path = '/home/ypj/dataset/cholec80_desmoking/psv2rs/light/train/0.bmp'

img = plt.imread(img_path)
plt.imshow(img)
plt.show()