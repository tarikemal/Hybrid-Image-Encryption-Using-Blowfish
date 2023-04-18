1. ABOUT THE PROJECT

- This project encrypts given images with a hybrid technique I developed.
  - Image is first encrypted with my encryption algorithm with the key generated by heuristic algorithms.
  - Then the output is given to the blowfish encryption algorithm to get the final encrypted version of the original image.

- Encrypted images generated by different heuristic algorithms(GA/SA/HC) are then compared by using SSIM and PSNR units.
- Encrypted images generated by different hybrid algorithms(GA-Blow, SA-Blow, HC-Blow) are also compared by using SSIM and PSNR units.

2. BUILT WITH

- Python 3.10.5 was used for the entire project. No additional frameworks/plugins were used.

3. GETTING STARTED

   3.1. PREREQUISITES

    - You should run setup.py to install necessary packages to run the project.

  3.2. RUNNING THE PROJECT

    - After installing the necessary packages with setup.py, you should run main.py file to start the project.
    - Running main.py, user will get a prompt asking for the comparison unit to compare the encrypted images with the original images. These comparison results will         later be used to create comparison tables.
    - User should enter ssim or psnr as input. After taking the input from user, program will run necessary encryption functions (explained in detail by commentary on       main.py) and draw the comparison tables for heuristic, hybrid encryption methods. It will also draw a table that compares traditional Blowfish encryption algorithm       with the hybrid encryption methods.
    - Program saves the images which are encrytped by blowfish and our hybrid technique under the "encrypted-images/blowfish" directory. Images that are encrypted using     only heuristic algorithms are saved under the path according to the heuristic method used.
    (ex. "encrypted-images/ga", "encrypted-images/sa")
