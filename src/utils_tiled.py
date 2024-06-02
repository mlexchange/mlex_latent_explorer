from tiled.client import from_uri


class TiledResults:
    def __init__(self, tiled_uri, tiled_api_key=None):
        self.tiled_uri = tiled_uri
        self.tiled_api_key = tiled_api_key
        self.tiled_client = from_uri(tiled_uri, api_key=tiled_api_key)
        self.prep_result_tiled_containers()

    def prep_result_tiled_containers(self):
        container = self.tiled_client
        if "latent_vectors" not in container.keys():
            container.create_container(key="latent_vectors")
        pass

    def get_metadata(self, flow_run_id):
        latent_vectors = self.tiled_client["latent_vectors"]
        if flow_run_id in latent_vectors:
            container = latent_vectors[flow_run_id]
            metadata = container.metadata
        else:
            metadata = None
        return metadata

    def get_latent_vectors(self, flow_run_id):
        latent_vectors = self.tiled_client["latent_vectors"]
        if flow_run_id in latent_vectors:
            container = latent_vectors[flow_run_id]
            latent_vectors = container.read()
        else:
            latent_vectors = None
        return latent_vectors
