# 用 nibabel 加载 nii 文件
nii_file = '/Users/pangxiaolong/Downloads/DWI/0000000441.nii'  # 把这里的路径替换成你的文件路径

import matplotlib

matplotlib.use('TkAgg')

from matplotlib import pylab as plt

import nibabel as nib

from nibabel import nifti1

from nibabel.viewers import OrthoSlicer3D

example_filename = nii_file

img = nib.load(example_filename)

print (img)

print (img.header['db_name']) #输出头信息

width,height,queue=img.dataobj.shape

OrthoSlicer3D(img.dataobj).show()

num = 1

for i in range(0,queue,10):
    img_arr = img.dataobj[:,:,i]

plt.subplot(5,4,num)

plt.imshow(img_arr,cmap='gray')

num +=1

plt.show()
