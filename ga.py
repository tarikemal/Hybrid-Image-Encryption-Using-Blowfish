import numpy as np
import random

# Calculate Fitness
def calcFitness(individual):
  # Doesn't work if not numpy array
  individualArr = np.array(individual) 

  fitness = 0
  individualLength = len(individual)

  for i in range(individualLength):
    nextIndex = np.where(individualArr == i)[0]

    # if array is empty, it adds nothing to the fitness value
    if nextIndex.size == 0:
      index = 0

    # else, add up all the numbers inside the array. 
    else:
      count = nextIndex.size
      nextIndex = sum(nextIndex)
      # Substract the index by the repetition count
      index = abs(nextIndex - i * count)

    fitness = fitness + index

  return fitness

# Perform a single-point crossover
def crossover(parent1, parent2, individualLength):
  index = random.randint(1, individualLength-1)

  child1 = np.concatenate(
      [parent1[0:index], parent2[index:]]
  )
  child2 = np.concatenate(
      [parent2[0:index], parent1[index:]]
  )
  return child1, child2

# Perform an order one crossover
def orderOneCrossover(parent1, parent2, individualLength):
  left_index = random.randint(0, individualLength-1)
  right_index = random.randint(0, individualLength-1)
  if left_index > right_index:
    left_index, right_index = right_index, left_index

  child1 = [-1] * individualLength
  child2 = [-1] * individualLength

  for i in range(left_index, right_index + 1):
    child1[i] = parent1[i]
    child2[i] = parent2[i]

  child1_val = set(child1) #turn into sets to iterate
  child2_val = set(child2)

  for i, value in enumerate(parent1):
    if value not in child1_val:
      child1[child1.index(-1)] = value
      child1_val.add(value)
      
  for i, value in enumerate(parent2):
    if value not in child2_val:
      child2[child2.index(-1)] = value
      child2_val.add(value)

  return child1, child2

# Perform mutation on an individual that either increases or decreases a cell by one
def mutate(individual, mutation_probability):
  for i, gene in enumerate(individual):
    if random.random() < mutation_probability:
      if random.getrandbits(1) == 1:
        individual[i] = individual[i] + 1
      else:
        individual[i] = individual[i] - 1

  return individual

# Perform mutation on an individual that swaps two cells
def mutateSwap(individual, mutationProbability):
  if random.random() < mutationProbability:
    i = random.randint(0, len(individual) - 1)
    j = random.randint(0, len(individual) - 1)
    individual[i], individual[j] = individual[j], individual[i]

  return individual

# Create population
def createPopulation(popsize, lb, ub, individualLength):
    population = np.zeros((popsize, individualLength))
    for i in range(popsize):
        population[i] = random.sample(range(lb, ub), ub)
    return population

# Tournament selection that returns indexes
def tournamentSelectionID(scores, tournamentSize):
  tournament = random.choices(scores, k=tournamentSize)
  best = sorted(tournament)[-1]
  secondbest = sorted(tournament)[-2]
  index_best = np.where(scores == best)[0]
  index_secondbest = np.where(scores == secondbest)[0]
  return index_best, index_secondbest

# Roulette selection that returns indexes
def rouletteSelectionID(scores, popSize):
  reverse = max(scores) + min(scores)
  reverseScores = reverse - scores.copy()
  sumScores = sum(reverseScores)
  pick = random.uniform(0, sumScores)
  current = 0
  for individualId in range(popSize):
    current += reverseScores[individualId]
    if current > pick:
        return individualId

# Clear the duplicates in the population
def clearDupePopulation(population, lb, ub, individualLength):
  newPopulation = np.unique(population, axis = 0)
  oldLen = len(population)
  newLen = len(newPopulation)

  if newLen < oldLen :
    countDupes = oldLen - newLen
    popBatch = createPopulation(countDupes, lb, ub, individualLength)
    newPopulation = np.concatenate(newPopulation, popBatch)

  return newPopulation

# Selecting a pair to perform crossover on
def pairSelection(population, scores, popSize, selection):
    if selection == 'tournament':
      parent1Id, parent2Id = tournamentSelectionID(scores, 64)
      parent1 = population[parent1Id[0]].copy()
      parent2 = population[parent2Id[0]].copy()
      return parent1, parent2

    if selection == 'roulette':
      parent1Id = rouletteSelectionID(scores, popSize)
      parent2Id = rouletteSelectionID(scores, popSize)
      parent1 = population[parent1Id].copy()
      parent2 = population[parent2Id].copy()
      return parent1, parent2

# Calculate the scores
def calcScores(population, popsize):
    scores = np.zeros(popsize)
    individualLength = len(population[0])
    for i in range(popsize):
        scores[i] = calcFitness(population[i])
    return scores

# Fixing Duplicate Values That Occur From Crossover
def duplicateFix(lst):
  duplicate_indexes = []
  for i, elt in enumerate(lst):
    if elt in lst[i+1:]:
      duplicate_indexes.append(i)

  lost = []
  for i in range(len(lst)):
    if i not in lst:
      lost.append(i)
  for l in range(len(lost)):
    lst[duplicate_indexes[l]] = lost[l]
  
  return lst

# Perform an Ordered Crossover (Doesn't Work)
def orderedCrossover(parent1, parent2, individualLength):
    # Choose random start/end position for crossover
    child1, child2 = [-1] * individualLength, [-1] * individualLength
    start, end = sorted([random.randrange(individualLength) for _ in range(2)])

    child1_inherited = []
    child2_inherited = []
    for i in range(start, end + 1):
        child1[i] = parent1[i]
        child2[i] = parent2[i]
        child1_inherited.append(parent1[i])
        child2_inherited.append(parent2[i])

    #Fill the remaining position with the other parents' entries
    parent1_pos, parent2_pos = 0, 0

    fixed_pos = list(range(start, end + 1))       
    i = 0
    while i < individualLength:
        if i in fixed_pos:
            i += 1
            continue

        test_child1 = child1[i]
        if test_child1 == -1: #to be filled
            parent2_trait = parent2[parent2_pos] #index fazla geliyor niye anlamadım
            print("Parent2 trait: ", parent2_trait, " Parent2 pos: ", parent2_pos)
            while parent2_trait in child1_inherited:
                parent2_pos += 1
                parent2_trait = parent2[parent2_pos] 
            child1[i] = parent2_trait
            child1_inherited.append(parent2_trait)

        test_child2 = child2[i]
        if test_child2 == -1: #to be filled
            parent1_trait = parent1[parent2_pos]
            print("Parent1 trait: ", parent1_trait, " Parent1 pos: ", parent1_pos)
            while parent1_trait in child2_inherited:
                parent1_pos += 1
                parent1_trait = parent1[parent1_pos]
            child2[i] = parent1_trait
            child2_inherited.append(parent1_trait)
        i += 1

    return child1, child2

# Perform an Edge Recombination Crossover (Doesn't Work, infinite loop)
def ERX(parent1, parent2, individualLength):
    # Create empty offspring solutions
    child1 = [-1] * individualLength
    child2 = [-1] * individualLength

    # Create edge lists for each parent
    edges1 = []
    edges2 = []
    for i in range(individualLength - 1):
        edges1.append((parent1[i], parent1[i + 1]))
        edges2.append((parent2[i], parent2[i + 1]))
    edges1.append((parent1[individualLength - 1], parent1[0]))
    edges2.append((parent2[individualLength - 1], parent2[0]))

    # Add edges from one parent to the other until all values are present in one offspring
    used_edges1 = set()
    used_edges2 = set()
    child1_values = set()
    child2_values = set()
    counter = 0
    while True:
        if len(child1_values) < individualLength:
            unused_edges1 = [edge for edge in edges1 if edge not in used_edges1]
            if unused_edges1:
                edge = unused_edges1[random.randint(0, len(unused_edges1) - 1)]
                if edge[0] not in child1_values and edge[1] not in child1_values:
                    child1[child1.index(-1)] = edge[0]
                    child1[child1.index(-1)] = edge[1]
                    child1_values.add(edge[0])
                    child1_values.add(edge[1])
                    used_edges1.add(edge)
                    used_edges2.add(edge)
                    counter += 1
        if len(child2_values) < individualLength:
            unused_edges2 = [edge for edge in edges2 if edge not in used_edges2]
            if unused_edges2:
                edge = unused_edges2[random.randint(0, len(unused_edges2) - 1)]
                if edge[0] not in child2_values and edge[1] not in child2_values:
                    child2[child2.index(-1)] = edge[0]
                    child2[child2.index(-1)] = edge[1]
                    child2_values.add(edge[0])
                    child2_values.add(edge[1])
                    used_edges1.add(edge)
                    used_edges2.add(edge)
                    counter += 1
        if len(child1_values) == individualLength and len(child2_values) == individualLength:
            break

    return child1, child2

# Perform a Maximal Preservative Crossover
def MPX(parent1, parent2, individualLength):
    child1 = [-1] * individualLength
    child2 = [-1] * individualLength

    values1 = set(parent1)
    values2 = set(parent2)

    # Find common values between the two parents
    common_values = values1.intersection(values2)
    pivot = list(common_values)[random.randint(0, len(common_values) - 1)]

    # Find indices of pivot in both parents
    pivot_index1 = list(parent1).index(pivot)
    pivot_index2 = list(parent2).index(pivot)

    # Assign pivot value to offspring solutions
    child1[pivot_index1] = pivot
    child2[pivot_index2] = pivot

    # Assign remaining values to offspring solutions
    for i in range(individualLength):
        if child1[i] == -1:
            child1[i] = parent1[i]
        if child2[i] == -1:
            child2[i] = parent2[i]

    return child1, child2

def PMX(parent1, parent2, individualLength):
  left_index, right_index = sorted([random.randrange(individualLength) for _ in range(2)])

  child1 = [-1] * individualLength
  child2 = [-1] * individualLength

  for i in range(left_index, right_index + 1):
    child1[i] = parent1[i]
    child2[i] = parent2[i]

  child1_values = [value for i, value in enumerate(parent2) if i < left_index or i > right_index]
  child2_values = [value for i, value in enumerate(parent1) if i < left_index or i > right_index]

  for i, value in enumerate(child1):
    if value == -1:
      child1[i] = child1_values.pop(0)

  for i, value in enumerate(child2):
    if value == -1:
      child2[i] = child2_values.pop(0)

  return child1, child2

def geneticAlgorithm(population_size, tournament_size, mutation_probability, max_generations, lb, ub, individualLength, selection, crossoverFunc):
  population = createPopulation(population_size, lb, ub, individualLength)
  population = clearDupePopulation(population, lb, ub, individualLength)
  scores = calcScores(population, population_size)
  population = np.array(population)
  for generation in range(max_generations):
    # print("Iteration: ", generation + 1)
    new_population = []
    for _ in range(population_size):
      parent1, parent2 = pairSelection(population, scores, population_size, selection)
      if crossoverFunc == 1:
        child1, child2 = crossover(parent1, parent2, individualLength)
      elif crossoverFunc == 2:
        child1, child2 = orderOneCrossover(parent1, parent2, individualLength)
      elif crossoverFunc == 3:
        child1, child2 = orderedCrossover(parent1, parent2, individualLength)
      elif crossoverFunc == 4:
        child1, child2 = ERX(parent1, parent2, individualLength)
      elif crossoverFunc == 5:
        child1, child2 = MPX(parent1, parent2, individualLength)
      elif crossoverFunc == 6:
        child1, child2 = PMX(parent1, parent2, individualLength)

      child1 = mutateSwap(child1, mutation_probability)
      child2 = mutateSwap(child2, mutation_probability)
      
      if crossoverFunc == 1:
        child1 = duplicateFix(child1)
        child2 = duplicateFix(child2)

      new_population.append(child1)
      new_population.append(child2)
    population = new_population
    key = max(population, key=calcFitness)
    fitness = calcFitness(key)
    
  #   print("Key: ", max(population, key=calcFitness))
  #   print("Fitness: ", fitness)
  # print("------------------")
  return max(population, key=calcFitness)