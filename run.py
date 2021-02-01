import random
import json
from deap import base, creator, tools
from scoop import futures
from functools import partial
from tqdm import tqdm


from operators import *
from logger import Logger

# force tqdm to use one line
tqdm = partial(tqdm, position=0, leave=True) 

SEED = 64
GEN_NUM = 25
POPULATION = 16
CXPB, MUTPB = 0.5, 0.2

class Algorithm:
    def __init__(self, try_concurrent=True):

        random.seed(SEED)

        self.logger = Logger()

        self.toolbox = base.Toolbox()
        self.__setup_individual()
        self.__setup_operators()       
        
        if try_concurrent:
            self.toolbox.register("map", futures.map)


    def __generation(self):
        offspring = self.__get_offspring()
        self.__apply_crossover(offspring)
        self.__apply_mutation(offspring)
        self.__evaluate_offspring(offspring)

        self.pop[:] = offspring
        

    def __apply_crossover(self, offspring):
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                self.toolbox.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values


    def __apply_mutation(self, offspring):
        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values


    def __evaluate_offspring(self, offspring):
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(tqdm(self.toolbox.map(self.toolbox.evaluate, invalid_ind),
            desc="Evaluation", total=len(invalid_ind)))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


    def __get_offspring(self):
        # Select the next generation individuals
        offspring = self.toolbox.select(self.pop, len(self.pop))
        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))
        
        return offspring

    def run(self):
        self.pop = self.toolbox.population(n=POPULATION)

        for generation_num in tqdm(range(GEN_NUM), desc="Generation", total=GEN_NUM):
            self.__generation()
            best_ind_gen = tools.selBest(self.pop, 1)[0]
            self.logger.log_generation(self.pop, best_ind_gen)

        
    def __setup_individual(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMax)
        
        self.toolbox.register("individual", random_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def __setup_operators(self):
        self.toolbox.register("evaluate", evaluateModel)
        self.toolbox.register("mate", cxTwoPoint_op)
        self.toolbox.register(
            "mutate", mutate_op, 
            low={
                "gamma": 0.9,
                "learning_rate": 0.000001
            }, 
            up={
                "gamma": 1,
                "learning_rate": 0.01
            })

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        self.toolbox.register("select", tools.selTournament, tournsize=3)


a = Algorithm()
if __name__ == "__main__":
    a.run()