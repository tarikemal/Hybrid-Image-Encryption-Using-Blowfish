import random
from PIL import Image
from encrypt import encryptImage
from sa import simulatedAnnealing
from hc import hillClimbing
from ga import geneticAlgorithm
from compare import compareImagesWithSSIM, compareImagesWithPSNR
import os
from blowfish import encryptImageByBlowfish

# Resizes original images to compare them with encrypted(320x320) images later on
def resizeOriginalImages():
    directory = "original-images"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        image = Image.open(f)
        img_resized = image.resize((320, 320))
        img_resized.save(f)

# Prints table where heuristic encryptions are compared
def createAlgorithmComparisonTable(randomRes, hcRes, saRes, gaRes):
    print("----------------------------------------------------")
    print("            Random       HC         SA         GA")
    print("----------------------------------------------------")
    rows = []
    for imgRandom in randomRes["algo"]:
        rows.append([imgRandom[0], imgRandom[1]])
    cnt = 0
    for imgHC in hcRes["algo"]:
        rows[cnt].append(imgHC[1])
        cnt += 1
    cnt = 0
    for imgSA in saRes["algo"]:
        rows[cnt].append(imgSA[1])
        cnt += 1
    cnt = 0
    for imgGA in gaRes["algo"]:
        rows[cnt].append(imgGA[1])
        cnt += 1
            
    for row in rows:
        print(row[0] + " "*(10-len(row[0])) + "|" + " " + str(row[1]) + " "*(11-len(str(row[1]))) + str(row[2]) + " "*(11-len(str(row[2]))) + str(row[3]) +  " "*(11-len(str(row[3]))) + str(row[4]))

# Prints table where hybrid encryptions are compared
def createHybridAlgorithmComparisonTable(randomRes, hcRes, saRes, gaRes):   
    print("----------------------------------------------------------------------")
    print("            Random-Blow       HC-Blow         SA-Blow         GA-Blow")
    print("----------------------------------------------------------------------")
    rows = []
    for imgRandom in randomRes["hybrid"]:
        rows.append([imgRandom[0], imgRandom[1]])
    cnt = 0
    for imgHC in hcRes["hybrid"]:
        rows[cnt].append(imgHC[1])
        cnt += 1
    cnt = 0
    for imgSA in saRes["hybrid"]:
        rows[cnt].append(imgSA[1])
        cnt += 1
    cnt = 0
    for imgGA in gaRes["hybrid"]:
        rows[cnt].append(imgGA[1])
        cnt += 1
            
    for row in rows:
        print(row[0] + " "*(10-len(row[0])) + "|" + " "*3 + str(row[1]) + " "*(16-len(str(row[1]))) + str(row[2]) + " "*(16-len(str(row[2]))) + str(row[3]) +  " "*(16-len(str(row[3]))) + str(row[4]))

# Prints table where blowfish algorihm and hybrid encryptions are compared
def createHybridVsTraditionalTable(randomRes, hcRes, saRes, gaRes):
    print("-------------------------------------------------------------------------------------")
    print("            Blowfish       Random-Blow       HC-Blow         SA-Blow         GA-Blow")
    print("-------------------------------------------------------------------------------------")
    rows = []
    for imgBlow in randomRes["blow"]:
        rows.append([imgBlow[0], imgBlow[1]])
    cnt = 0
    for imgRandom in randomRes["hybrid"]:
        rows[cnt].append(imgRandom[1])
        cnt += 1
    cnt = 0
    for imgHC in hcRes["hybrid"]:
        rows[cnt].append(imgHC[1])
        cnt += 1
    cnt = 0
    for imgSA in saRes["hybrid"]:
        rows[cnt].append(imgSA[1])
        cnt += 1
    cnt = 0
    for imgGA in gaRes["hybrid"]:
        rows[cnt].append(imgGA[1])
        cnt += 1
    # print(rows)        
    for row in rows:
        print(row[0] + " "*(10-len(row[0])) + "|" + " "*2 + str(row[1]) + " "*(16-len(str(row[1]))) + str(row[2]) + " "*(16-len(str(row[2]))) + str(row[3]) +  " "*(16-len(str(row[3]))) + str(row[4]) + " "*(16-len(str(row[4]))) + str(row[5]))

# This function takes two arguments, heuristic algorithm to generate the key and comparison unit to compare the images.
# It iterates over the images in original-images directory. It encrypts every image 10 times, each time with a different key generated by the given heuristic algorithm.
# It takes the average of the comparison values from the 10 iterations.
# It returns a dictionary where comparison values of blowfish, heuristic and hybrid encrypted versions of the image hold for every image.          
def run(algo, comp_unit):
    counter = 10
    totalAlgo = 0
    totalHybrid = 0
    totalBlow = 0
    res = {"blow": [], "algo": [], "hybrid": []}
    key = []
    
    directory = "original-images"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        image = Image.open(f)
        
        for x in range(counter):
            if algo == "hc":
                key = hillClimbing(16)
            if algo == "sa":
                key = simulatedAnnealing(16)
            if algo == "ga":
                popsize = 100
                # lower bound
                lb = 0
                # upper bound
                ub = 16
                # individiual length
                individualLength = 16
                # tournament size
                tsize = 10
                # maximum generations
                maxgen = 100
                # mutation probability
                mutprob = 0.1
                # selection method
                selection = 'tournament'
                # crossover method { 1: Single-Point, 2: Order-One, 3: Ordered, 4: ERX, 5: MPX, 6: PMX}  -- 3 ve 4 çalışmıyor
                crossoverFunc = 2
                # generate key
                key = geneticAlgorithm(popsize, tsize, mutprob, maxgen, lb, ub, individualLength, selection, crossoverFunc)
            if algo == "random":
                key = random.sample(range(0, 16), 16)  
                           
            blow_image = encryptImageByBlowfish(f)                                  # Path of image encrypted by only blowfish algorithm
            algo_encrypted_image = encryptImage(image, key, f, algo)                # Path of image encrypted by only heuristic algorithm 
            hybrid_encryted_image = encryptImageByBlowfish(algo_encrypted_image)    # Path of image encrypted by our hybrid technique
        
            if comp_unit == "psnr":  
                compBlow   = compareImagesWithPSNR(f, blow_image)
                compAlgo   = compareImagesWithPSNR(f, algo_encrypted_image)
                compHybrid = compareImagesWithPSNR(f, hybrid_encryted_image)
            elif comp_unit == "ssim":
                compBlow   = compareImagesWithSSIM(f, blow_image)
                compAlgo   = compareImagesWithSSIM(f, algo_encrypted_image)
                compHybrid = compareImagesWithSSIM(f, hybrid_encryted_image)
            else:
                print("Please give a valid comparison unit(ssim or psnr)!")    
                return
            
            totalBlow += compBlow
            totalAlgo += compAlgo
            totalHybrid += compHybrid
        
        totalBlow /= counter
        totalAlgo /= counter
        totalHybrid /= counter
        
        res["blow"].append((filename.split('.')[0], round(totalBlow, 4)))
        res["algo"].append((filename.split('.')[0], round(totalAlgo, 4)))
        res["hybrid"].append((filename.split('.')[0],round(totalHybrid, 4)))
    return res
        
### Program Starts Here ###

resizeOriginalImages()

comp_unit = input("Enter the unit to use during comparison of images (psnr or ssim): ") # ssim or psnr

print("\nRunning blowfish, heuristic, hybrid encryption algorithms with different heuristic methods on all original images...")
randomRes = run("random", comp_unit)
hillClimbingRes = run("hc", comp_unit)
simulatedAnnealingRes = run("sa", comp_unit)
geneticAlgorithmRes = run("ga", comp_unit)

# randomRes = {'blow': [('kite', 9.6622), ('frog', 9.6911), ('peppers', 9.7451), ('drawing', 8.7662), ('telephone', 11.175)], 'algo': [('kite', 6.5023), ('frog', 5.0997), ('peppers', 5.4282), ('drawing', 3.9174), ('telephone', 8.0397)], 'hybrid': [('kite', 9.804), ('frog', 9.6194), ('peppers', 9.7495), ('drawing', 8.8594), ('telephone', 11.2724)]}
# hillClimbingRes = {'blow': [('kite', 9.6622), ('frog', 9.6911), ('peppers', 9.7451), ('drawing', 8.7662), ('telephone', 11.175)], 'algo': [('kite', 6.5774), ('frog', 5.1107), ('peppers', 5.4545), ('drawing', 3.9341), ('telephone', 8.1299)], 'hybrid': [('kite', 9.7895), ('frog', 9.6101), ('peppers', 9.7421), ('drawing', 8.8596), ('telephone', 11.2864)]}
# simulatedAnnealingRes = {'blow': [('kite', 9.6622), ('frog', 9.6911), ('peppers', 9.7451), ('drawing', 8.7662), ('telephone', 11.175)], 'algo': [('kite', 6.5785), ('frog', 5.1129), ('peppers', 5.4628), ('drawing', 3.9338), ('telephone', 8.1336)], 'hybrid': [('kite', 9.7941), ('frog', 9.6117), ('peppers', 9.7439), ('drawing', 8.8307), ('telephone', 11.2578)]}
# geneticAlgorithmRes = {'blow': [('kite', 9.6622), ('frog', 9.6911), ('peppers', 9.7451), ('drawing', 8.7662), ('telephone', 11.175)], 'algo': [('kite', 6.5543), ('frog', 5.1085), ('peppers', 5.4608), ('drawing', 3.9349), ('telephone', 8.1233)], 'hybrid': [('kite', 9.7931), ('frog', 9.6154), ('peppers', 9.7518), ('drawing', 8.8499), ('telephone', 11.2752)]}

print("\nDrawing the comparison tables...")
print("\n\n")
createAlgorithmComparisonTable(randomRes, hillClimbingRes, simulatedAnnealingRes, geneticAlgorithmRes)
print("\n\n")
createHybridAlgorithmComparisonTable(randomRes, hillClimbingRes, simulatedAnnealingRes, geneticAlgorithmRes)
print("\n\n")
createHybridVsTraditionalTable(randomRes, hillClimbingRes, simulatedAnnealingRes, geneticAlgorithmRes)

