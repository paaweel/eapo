import random
import json
from deap import base, creator, tools
from scoop import futures
from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True)

from operators import crossover_op, mutate_op, evaluateModel, random_individual


SEED = 64
CXPB, MUTPB = 0.5, 0.2

class Algorithm:
    def __init__(self, try_concurrent=True):

        random.seed(SEED)

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
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # print("  Evaluated %i individuals" % len(invalid_ind))


    def __get_offspring(self):
        # Select the next generation individuals
        offspring = self.toolbox.select(self.pop, len(self.pop))
        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))
        
        return offspring


    def __log(self):
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)


    def run(self):

        self.pop = self.toolbox.population(n=12)

        # pbar = tqdm(range(50))
        for generation_num in tqdm(range(50)):
            self.__generation()
            # pbar.set_description("Processing gen: %s" % generation_num)

            # best_ind_gen = tools.selBest(pop, 1)[0]

            # data.append({"generation": g, "parameters": best_ind_gen, "fitness": best_ind_gen.fitness.values})
            # with open(loggFile, 'w') as f:
            #     json.dump(data, f)

        
    def __setup_individual(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMax)
        
        self.toolbox.register("individual", random_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def __setup_operators(self):
        self.toolbox.register("evaluate", evaluateModel)
        self.toolbox.register("mate", crossover_op)
        self.toolbox.register(
            "mutate", mutate_op, 
            low={
                "gamma": 0,
                "learning_rate": 0
            }, 
            up={
                "gamma": 1,
                "learning_rate": 1
            })

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        self.toolbox.register("select", tools.selTournament, tournsize=3)


if __name__ == "__main__":
    a = Algorithm()
    a.run()