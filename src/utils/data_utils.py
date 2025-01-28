import hashlib
import os

import httpx
from humanhash import humanize
from tiled.client import from_uri

RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")


class TiledDataLoader:
    def __init__(self, data_tiled_uri, data_tiled_api_key):
        self.data_tiled_uri = data_tiled_uri
        self.data_tiled_api_key = data_tiled_api_key
        self.data_client = from_uri(
            self.data_tiled_uri,
            api_key=self.data_tiled_api_key,
            timeout=httpx.Timeout(30.0),
        )

    def refresh_data_client(self):
        self.data_client = from_uri(
            self.data_tiled_uri,
            api_key=self.data_tiled_api_key,
            timeout=httpx.Timeout(30.0),
        )

    def prepare_project_container(self, user, project_name):
        """
        Prepare a project container in the data store
        """
        last_container = self.data_client
        for part in [user, project_name]:
            if part in last_container.keys():
                last_container = last_container[part]
            else:
                last_container = last_container.create_container(key=part)
        return last_container

    def get_data_by_trimmed_uri(self, trimmed_uri, slice=None):
        """
        Retrieve data by a trimmed uri (not containing the base uri) and slice id
        """
        if slice is None:
            return self.data_client[trimmed_uri]
        else:
            return self.data_client[trimmed_uri][slice]

    def get_metadata_by_trimmed_uri(self, trimmed_uri):
        """
        Retrieve metadata by a trimmed uri (not containing the base uri)
        """
        return self.data_client[trimmed_uri].metadata


tiled_results = TiledDataLoader(
    data_tiled_uri=RESULTS_TILED_URI, data_tiled_api_key=RESULTS_TILED_API_KEY
)


def hash_list_of_strings(strings_list):
    """
    Produces a hash of a list of strings.
    """
    concatenated = "".join(strings_list)
    digest = hashlib.sha256(concatenated.encode("utf-8")).hexdigest()
    return humanize(digest)


def get_input_params(children):
    """
    Gets the model parameters and its corresponding values
    """
    input_params = {}
    if bool(children):
        try:
            for child in children["props"]["children"]:
                key = child["props"]["children"][1]["props"]["id"]["param_key"]
                value = child["props"]["children"][1]["props"]["value"]
                input_params[key] = value
        except Exception:
            for child in children:
                key = child["props"]["children"][1]["props"]["id"]
                value = child["props"]["children"][1]["props"]["value"]
                input_params[key] = value
    return input_params
