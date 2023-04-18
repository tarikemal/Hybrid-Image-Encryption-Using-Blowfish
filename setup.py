import sys
import subprocess

# implement pip as a subprocess:
packages = ["Pillow", "pycryptodome", "numpy", "opencv-python", "scikit-image", "patchify"]
for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
    package])