import os

folders = [
    r"D:\DnadefectDetectionproj\datasets\test\defective",
    r"D:\DnadefectDetectionproj\datasets\test\normal",

]

for folder in folders:
    if os.path.isdir(folder):
        print(folder, "has", len(os.listdir(folder)), "images")
    else:
        print(folder, "is not a folder")
