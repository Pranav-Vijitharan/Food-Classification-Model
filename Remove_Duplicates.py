from imutils import paths
import numpy as np
import argparse
import cv2
import os

def dhash(image, hashSize=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="Path to Dataset")
ap.add_argument("-r", "--remove", type=int, default=-1,
    help="whether or not duplicates should be removed (i.e., dry run)")
args = vars(ap.parse_args())
print("[INFO] computing image hashes...")
imagePaths = list(paths.list_images(args["dataset"]))

hashes = {}
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    h = dhash(image)
    p = hashes.get(h, [])
    p.append(imagePath)
    hashes[h] = p

for (h, hashedPaths) in hashes.items():
    if len(hashedPaths) > 1:
        if args["remove"] <= 0:
            print("[INFO] Hash:", h)
            print("Duplicate images:")
            for p in hashedPaths:
                print("- File:", p)
                image = cv2.imread(p)
                image = cv2.resize(image, (150, 150))
                cv2.imshow("Duplicate Image", image)
                cv2.waitKey(0)
        else:
            for p in hashedPaths[1:]:
                os.remove(p)


