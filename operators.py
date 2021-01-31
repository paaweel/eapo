import random
import struct
import math
import torch as th

from model import Model

# --- crossover operators -------------------
def crossover_op(ind1, ind2):
    better_ind = ind1 if ind1.fitness > ind2.fitness else ind2
    for key in ind1.keys():
        ind1[key] = ind1[key] + ind2[key]
        ind1[key] /= 2.0

    return ind1, better_ind
    

def cxTwoPoint_op(ind1, ind2):
    """ Two point crossover on the input"""
    size = len(ind1.keys())
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)

    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    for param in list(ind1.keys())[cxpoint1:cxpoint2]:
        ind1[param] = ind2[param]

    return ind1, ind2


def cxOnePoint_op(ind1, ind2):
    """ One point crossover on the input"""
    size = min(len(ind1.keys()), len(ind2))
    cxpoint = random.randint(1, size - 1)
    lhs_params = list(ind1.keys())[:cxpoint]
    rhs_params = list(ind1.keys())[cxpoint:]

    for param in lhs_params:
        ind1[param] = ind2[param]

    for param in rhs_params:
        ind2[param] = ind1[param]

    return ind1, ind2

# --- mutation operators -------------------

def mutate_op(ind1, up, low):
    number_of_params = random.randint(1, len(ind1))
    choosen_list = random.sample(list(ind1.keys()), number_of_params)
    for choosen in choosen_list:
        if choosen == "net_arch":
            ind1[choosen] = mutate_policy_kwargs(ind1[choosen])
        else:
            percentage = 0.02
            temp = ind1[choosen] * random.choice([1 - percentage, 1 + percentage])
            ind1[choosen] = clamp(temp, low[choosen], up[choosen])
    return ind1

def mutate_policy_kwargs(net_arch):
    size = len(net_arch)
    neurons = net_arch[0]

    n_layers = clamp(size + random.choice([-1, 0, 1]), 1, 5)
    width = math.log(neurons, 2)
    width = clamp(width + random.choice([-1, 0, 1]), 6, 9) ** 2

    net_arch = [int(width)] * n_layers

    return net_arch


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


# the goal ('fitness'), function to be maximized
def evaluateModel(individual):
    model = Model(individual)
    model.learn(50_000)
    return model.evaluate(),


def random_individual(ctor: callable) -> dict:
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
    paramters["gamma"] = random.uniform(0, 1)
    paramters["learning_rate"] = random.uniform(0.000001, 0.01)
    paramters["net_arch"] = get_net_arch()

    # paramters["learning_starts"] = 500 # random.randrange(100, 3000)
    # paramters["n_timesteps"] = random.randrange(1000, 3_000_000)
    # # paramters["policy"] = 'MlpPolicy'
    # # paramters["model_class"] = 'sac'
    # paramters["n_sampled_goal"] = random.randrange(1, 5)
    # # paramters["goal_selection_strategy"] = 'future'
    # paramters["buffer_size"] = random.randrange(1000, 30000)S
    # # paramters["ent_coef"] = 'auto'
    # paramters["batch_size"] = random.randrange(1, 4096)

    # paramters["online_sampling"] = bool(random.getrandbits(1))
    # paramters["normalize"] = bool(random.getrandbits(1))

    return ctor(paramters)


def get_net_arch():
    """
        https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
    """
    n_layers = random.randint(1,5)
    width = 2 ** random.randint(6, 9)
    return [int(width)] * n_layers


if __name__ == "__main__":
    def identity(x):
        return x


    i1 = random_individual(identity)

    print("mutation")
    print(i1)
    print(mutate_op(i1))

    print("mating")
    i1 = random_individual(identity)
    i2 = random_individual(identity)

    print("Individual 1")
    print(i1)

    print("Individual 2")
    print(i2)

    print("Child")
    print(crossover_op(i1, i2))
