import json

class Logger():
    def __init__(self):
        super().__init__()

        self.output_dir = "./data/"

        self.data = {
            "best_individual": {
                "parameters": "",
                "fitness": -1
            },
            "generations": []
        }


    def save_to_file(self):
        with open(self.output_dir + "logs.json", 'w') as f:
            json.dump(self.data, f, indent=4)


    def log_generation(self, population, best_ind):
        # log best individual for every generation
        gen = {
            "best_individual": {
                "parameters": best_ind,
                "fitness": best_ind.fitness.values[0]
            },
            "stats": self.stats(population)
        }
        self.data["generations"].append(gen)

        # update best individual (solution)
        if self.data["best_individual"]["fitness"] < best_ind.fitness.values[0]:
            self.data["best_individual"] = {
                "parameters": best_ind,
                "fitness": best_ind.fitness.values[0]
            }

        self.save_to_file()

    def stats(self, population):
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        return {
            "min": min(fits),
            "max": max(fits),
            "avg": mean,
            "std": std
        }