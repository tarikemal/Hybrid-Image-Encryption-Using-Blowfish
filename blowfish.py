from Crypto.Cipher import Blowfish
from PIL import Image

def parseString(string, stop_char):
    """Parses the string from the end and stops when it reaches the given character"""
    parsed_string = ""
    for char in reversed(string):
        if char == stop_char:
            break
        parsed_string += char
    return parsed_string[::-1]

def encryptImageByBlowfish(fname):
    # Open the image file
    im = Image.open(fname)

    # Read the image data as a bytes object
    data = im.tobytes()

    key = b'Key of length 16'
    cipher = Blowfish.new(key, Blowfish.MODE_ECB)

    # Encrypt the image data
    encrypted_data = cipher.encrypt(data)
    im = Image.frombytes('RGB', im.size, encrypted_data)
    
    output_path = parseString(fname, "/")
    output_path = "encrypted-images/blowfish/" + "blow" + "-" + output_path
    im.save(output_path)
    
    return output_path
    