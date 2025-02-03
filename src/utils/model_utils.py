import json


class Models:
    def __init__(self, modelfile_path="./assets/models.json", model_type=None):
        self.path = modelfile_path
        f = open(self.path)
        contents = json.load(f)["contents"]
        self.models = {}
        if model_type:
            self.modelname_list = []
            for content in contents:
                if content["type"] == model_type:
                    model_name = content["model_name"]
                    self.modelname_list.append(model_name)
                    self.models[model_name] = content
        else:
            self.modelname_list = [content["model_name"] for content in contents]
            for i, n in enumerate(self.modelname_list):
                self.models[n] = contents[i]

    def __getitem__(self, key):
        try:
            return self.models[key]
        except KeyError:
            raise KeyError(f"A model with name {key} does not exist.")
