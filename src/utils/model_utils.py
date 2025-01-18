import json


class Models:
    def __init__(self, modelfile_path="./assets/models.json"):
        self.path = modelfile_path
        f = open(self.path)
        contents = json.load(f)["contents"]
        self.modelname_list = [content["model_name"] for content in contents]
        self.models = {}

        for i, n in enumerate(self.modelname_list):
            self.models[n] = contents[i]

    def __getitem__(self, key):
        try:
            return self.models[key]
        except KeyError:
            raise KeyError(f"A model with name {key} does not exist.")
