import numpy as np
import dxchange
import glob, os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

prefix = 'output_multislice_constant_alpha_blur_20_rep*'
folder_ls = glob.glob(prefix)
folder_ls.sort()

fig, axes = plt.subplots(len(folder_ls) + 1, 2, figsize=(5, (len(folder_ls) + 1) * 5))
img = np.squeeze(dxchange.read_tiff(os.path.join(folder_ls[0], 'input1_original.tiff')))
axes[0, 0].imshow(img)
img = np.squeeze(dxchange.read_tiff(os.path.join(folder_ls[0], 'input2_original.tiff')))
axes[0, 1].imshow(img)

for i, folder in enumerate(folder_ls):
    img = dxchange.read_tiff(os.path.join(folder, 'reflection_9999.tiff'))
    axes[i + 1, 0].imshow(img)
    img = dxchange.read_tiff(os.path.join(folder, 'transmission_9999.tiff'))
    axes[i + 1, 1].imshow(img)
plt.show()