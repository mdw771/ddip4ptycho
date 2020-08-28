import matplotlib.pyplot as plt
import numpy as np
import dxchange
import os

def normalize(img):
    img = (img - np.mean(img)) / np.std(img)
    img = (img - img.min()) / (img.max() - img.min())
    return img

folder_ls = ['output_multislice_constant_alpha_1',
             'output_multislice_constant_alpha_10',
             'output_multislice_constant_alpha_blur_1',
             'output_multislice_constant_alpha_blur_10',]

fig, axes = plt.subplots(len(folder_ls) + 1, 2, figsize=(8, 5*(len(folder_ls) + 1)))

img = dxchange.read_tiff('data/au_ni.tiff')
img = img[114:114+272, 114:-114, :]
img = np.transpose(img, (2, 0, 1))
img = normalize(img)
std, mean = np.std(img[0]), np.mean(img[0])
axes[0, 0].imshow(img[0], cmap='gray', vmin=mean - 3 * std, vmax=mean + 3 * std)
std, mean = np.std(img[1]), np.mean(img[1])
axes[0, 1].imshow(img[1], cmap='gray', vmin=mean - 3 * std, vmax=mean + 3 * std)
axes[0, 0].set_title('Original')

for i, folder in enumerate(folder_ls):
    img = dxchange.read_tiff(os.path.join(folder, 'reflection_6999.tiff'))
    std, mean = np.std(img), np.mean(img)
    axes[i+1, 0].imshow(img, cmap='gray', vmin=mean - 3 * std, vmax=mean + 3 * std)
    axes[i+1, 0].set_title(folder)

    img = dxchange.read_tiff(os.path.join(folder, 'transmission_6999.tiff'))
    std, mean = np.std(img), np.mean(img)
    axes[i+1, 1].imshow(img, cmap='gray', vmin=mean - 3 * std, vmax=mean + 3 * std)
plt.show()
