import random

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


def random_individual(ctor) -> dict:
    """
        Brief: 
            Returns random individual

        Note:
            - Algorithm will not break if some of the params are not set
                Model class will use its own defaults

            - bool(random.getrandbits(1)) seems to be the fastest way of
                getting random bool value

    """
    paramters = {}

    # paramters["n_timesteps"] = random.randrange(1000, 30000)
    # # paramters["policy"] = 'MlpPolicy'
    # # paramters["model_class"] = 'sac'
    # paramters["n_sampled_goal"] = random.randrange(1, 5)
    # # paramters["goal_selection_strategy"] = 'future'
    # paramters["buffer_size"] = random.randrange(1000, 30000)
    # # paramters["ent_coef"] = 'auto'
    # paramters["batch_size"] = random.randrange(1, 4096)
    # paramters["gamma"] = random.uniform(0, 1)
    # paramters["learning_rate"] = random.uniform(0, 1)
    # paramters["learning_starts"] = random.uniform(0, 1)
    # paramters["online_sampling"] = bool(random.getrandbits(1))
    paramters["normalize"] = bool(random.getrandbits(1))

    return ctor(paramters)