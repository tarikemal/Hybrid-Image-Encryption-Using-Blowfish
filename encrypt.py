from patchify import patchify, unpatchify
import numpy as np
from PIL import Image
import ga

# Image encryption algorithm using patchify
def encryptImage(img, key, file_name, algo):
  img_resized = img.resize((320, 320))
  img_resized_arr = np.asarray(img_resized) 

  image_height, image_width, channel_count = img_resized_arr.shape
  patch_height, patch_width, step = 16, 16, 16
  patch_shape = (patch_height, patch_width, channel_count)

  patches = patchify(img_resized_arr, patch_shape, step=step) 
  
  for k in range(len(key)):
    if k not in key:
      key = ga.duplicateFix(key)
      key = key.tolist()

  encrypted_patches = np.zeros(patches.shape, dtype=int)
  for i in range(patches.shape[0]): 
    for j in range(patches.shape[1]): 
      for n in range(patches.shape[3]): 
        for k in range(len(key)):
          index = key.index(k) 
          patch = patches[i][j][0][n][k] 
          encrypted_patches[i][j][0][n][index] = patch 

  for i in range(patches.shape[0]): 
    for j in range(patches.shape[1]): 
      for n in range(patches.shape[4]): 
        for k in range(len(key)):
          try:
            index = key.index(k) 
            patch = encrypted_patches[i][j][0][k][n] 
            encrypted_patches[i][j][0][index][n] = patch 
          except ValueError:
            pass

  output_height = image_height - (image_height - patch_height) % step
  output_width = image_width - (image_width - patch_width) % step
  output_shape = (output_height, output_width, channel_count)

  output_image = unpatchify(encrypted_patches, output_shape)
  output_image = Image.fromarray((output_image * 255).astype(np.uint8))
  
  output_path = 'encrypted-images/' + algo + '/' + algo + '-'
  output_path += file_name[16:]
  
  if '.jpg' in file_name:
    im1 = output_image.save(output_path)
  elif '.png' in file_name:
    im1 = output_image.save(output_path)

  return output_path