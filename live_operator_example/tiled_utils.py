from tiled.structures.data_source import Asset, DataSource
from tiled.structures.table import TableStructure


def write_results(
    write_client, latent_vectors, io_parameters, latent_vectors_path, metadata=None
):
    uid_save = io_parameters.uid_save

    # Save latent vectors to Tiled
    structure = TableStructure.from_pandas(latent_vectors)

    # Remove API keys from metadata
    if metadata:
        metadata["io_parameters"].pop("data_tiled_api_key", None)
        metadata["io_parameters"].pop("results_tiled_api_key", None)

    frame = write_client.new(
        structure_family="table",
        data_sources=[
            DataSource(
                structure_family="table",
                structure=structure,
                mimetype="application/x-parquet",
                assets=[
                    Asset(
                        data_uri=f"file://{latent_vectors_path}",
                        is_directory=False,
                        parameter="data_uris",
                        num=1,
                    )
                ],
            )
        ],
        metadata=metadata,
        key=uid_save,
    )

    frame.write(latent_vectors)
    pass
