import os
import math
import random

# Generates a random key by given length
def randomKey(length):
    digits = list(range(length))
    key = list()

    for i in range(length):
      randomDigit = digits[random.randint(0, len(digits) - 1)]
      key.append(randomDigit)
      digits.remove(randomDigit)
    return key

# Calculates the fitness of the key
def keyFitness(key):
    fitness = 0
    for i in range(len(key)):
      digitReplacement = abs(i - key.index(i))
      fitness += digitReplacement
    return fitness

# Returns random neighbours for a key
def getNeighbours(key):
    """Generates 10 arrays of 16 random integers between 0 and 16 (inclusive)"""
    neighbours = []
    cnt = 0
    while 1:
        neighbour = []
        while len(neighbour) < len(key):
          number = random.randint(0, len(key) - 1) 
          if number not in neighbour:
            neighbour.append(number)
        if neighbour in neighbours or neighbour == key:
          continue
        neighbours.append(neighbour)
        cnt+=1
        if cnt == 10: # Goes into infinite loop if the key space is smaller than 10 (like keys length of 3 => key space is 3! = 6 < 10)
          break
        if cnt >= math.factorial(len(key)): # Avoids infinite loop   
          print("Key space is too small. Terminating getNeighbours()...")
          break 
    return neighbours

# Returns the neighbour with the best fitness
def getBestNeighbour(neighbours):
    bestKeyFitness = keyFitness(neighbours[0])
    bestNeighbour = neighbours[0]
    for neighbour in neighbours:
        currentFitness = keyFitness(neighbour)
        if currentFitness > bestKeyFitness:
            bestKeyFitness = currentFitness
            bestNeighbour = neighbour
    return bestNeighbour

# Runs the simulated annealing algorithm and returns the solution
def simulatedAnnealing(keyLength, initialTemperature = 1.0, temperature_iterations = 10, final_temperature = 0.0001, alpha = 0.9):
    guess = randomKey(keyLength)
    best        = guess
    temperature = float(initialTemperature)
    fxBest = keyFitness(guess) # float("-inf")
    while (temperature > final_temperature): 
        for repeat in range(0, temperature_iterations):
            fxOld = keyFitness(guess)
            newGuess = random.choice(getNeighbours(guess))
            fxNew    = keyFitness(newGuess) 
            delta     = (fxNew - fxOld)
            r         = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            p         = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) 
            if (delta > 0 or r <= p):  
              guess = newGuess 
            if (fxNew > fxBest): 
              fxBest = fxNew
              best = guess
        temperature = alpha*temperature  
    return  best