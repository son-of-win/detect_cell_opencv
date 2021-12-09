import os
arr = os.listdir("C:\\Users\\vuot2\\OneDrive - lx36\\Documents\\image-processing\\images\\")
import glob
image_file = []
f = open("train_images.txt","w")
for file in glob.glob("*.png"):
    image_file.append(file)
    f.write(file[:-4] + "\n")
f.close()
