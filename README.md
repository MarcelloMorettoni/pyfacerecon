# Lightweight Face Recognition System on Raspberry Pi
Light face recognition in python
Check tutorial here: https://www.linkedin.com/pulse/say-cheese-building-custom-face-recognition-system-marcello-lnzoe/?trackingId=Hd%2Ff1UtJRe610r2xXgaGcQ%3D%3D

The goal of this tutorial is to create a lightweight face recognition system without the hassle of training complex models. While it won’t be perfect, it’s simple enough to run on devices like a Raspberry Pi! 😉

## Steps

### 1) Create a New Conda Environment

```bash
conda create --name face_recognition python=3.9
conda activate face_recognition
conda install -c conda-forge opencv=4.5.5
conda install -c conda-forge dlib=19.22
conda install -c conda-forge face_recognition=1.3.0
conda install numpy=1.21.5

```
### 2) Place photos of yourself
Collect a few photos of yourself and place them in a directory (e.g., images_of_me/). Make sure the images are clear and from different angles if possible. The more the better. It should work with three or four as minimum. Keep them close to the face and with a small size. 

### 3) run face_reck.py
Should open a window like this:

![Example Image](facerecon.gif)

You can type 'q' for quit.
