import random
import struct

from model import Model


def float2bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')


def bin2float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]


def crossover_op(ind1, ind2):
    for key in ind1.keys():
        ind1[key] = ind1[key] + ind2[key]
        ind1[key] /= 2.0

    return ind1


def mutate_op(ind1):
    choosen = random.choice(list(ind1.keys()))
    binary_rep = list(float2bin(ind1[choosen]))
    random_idx = random.randrange(len(binary_rep))
    binary_rep[random_idx] = "1" if binary_rep[random_idx] == "0" else "1"
    ind1[choosen] = bin2float("".join(binary_rep))
    return ind1


# the goal ('fitness') function to be maximized
def evaluateModel(individual):
    model = Model(individual)
    model.learn(2_000)
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
    paramters["gamma"] = random.uniform(0, 1)
    paramters["learning_rate"] = random.uniform(0, 1)

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
    
