from model import Model

def crossover_op(ind1, ind2):
    return ind1

def mutate_op(ind1):
    return ind1

# the goal ('fitness') function to be maximized
def evaluateModel(individual):
    model = Model(individual)
    model.learn(100)
    return model.evaluate(),
