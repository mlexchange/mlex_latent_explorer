# MLExchange Latent Space Explorer

An application for visualizing latent vectors in **2D or 3D** using **PCA** and **UMAP** for dimensionality reduction.

## Running as a Standalone Application (Using Docker)

The **Prefect server, Tiled server, the application, and the Prefect worker job** all run within a **single Docker container**. This eliminates the need to start the servers separately.

However, the **Prefect worker** must be run separately on your local machine (refer to step 5). 

## Steps to Run the Application

### 1 Clone the Repository

```sh
git clone https://github.com/mlexchange/mlex_latent_explorer.git
cd mlex_latent_explorer
```

### 2 Configure Environment Variables

Create a `.env` file using `.env.example` as a reference:

```sh
cp .env.example .env
```

Then **update the** `.env` file with the correct values.

**Important Note:** Due to the current tiled configuration, ensure that the `WRITE_DIR` is a subdirectory of the `READ_DIR` if the same tiled server is used for both reading data and writing results.

#### MLFlow Configuration in .env

When setting `MLFLOW_TRACKING_URI` in the `.env` file:

- If you run the [MLFlow server](https://github.com/xiaoyachong/mlex_mlflow) locally, you can set it to:
  ```
  MLFLOW_TRACKING_URI="http://mlflow-server:5000"
  ```
  This works because the MLFlow server also runs in the `mle_net` Docker network.

- If you run MLFlow server on vaughan and use SSH port forwarding:
  ```
  ssh -S forward -L 5000:localhost:5000 <your-username>@vaughan.als.lbl.gov
  ```
  Then you can set it to:
  ```
  MLFLOW_TRACKING_URI="http://host.docker.internal:5000"
  ```

### 3 Build and Start the Application

```sh
docker compose up -d
```

* `-d` → Runs the containers in the background (detached mode).

### 4 Verify Running Containers

```sh
docker ps
```

### 5 Start a Prefect Worker

Open another terminal and start a Prefect worker. Refer to [mlex_prefect_worker](https://github.com/mlexchange/mlex_prefect_worker) for detailed instructions on setting up and running the worker.


### 6 Access the Application

Once the container is running, open your browser and visit:
* **Dash app:** http://localhost:8070/

### 7 Stopping the Application

To stop and remove the running containers, use:

```sh
docker compose down
```

This will **shut down all services** but **retain data** if volumes are used.

## Model Description

For dimension reduction:
- [PCA](https://github.com/mlexchange/mlex_dimension_reduction_pca)

- [UMAP](https://github.com/mlexchange/mlex_dimension_reduction_umap)

For clustering:
- [KMeans, DBSCAN, HDBSCAN](https://github.com/mlexchange/mlex_clustering)

## Developer Setup
If you are developing this library, there are a few things to note.

1. Install development dependencies:

```
pip install .
pip install ".[dev]"
```

2. Install pre-commit
This step will setup the pre-commit package. After this, commits will get run against flake8, black, isort.

```
pre-commit install
```

3. (Optional) If you want to check what pre-commit would do before commiting, you can run:

```
pre-commit run --all-files
```

4. To run test cases:

```
python -m pytest
```


## Copyright
MLExchange Copyright (c) 2023, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches,
or upgrades to the features, functionality or performance of the source
code ("Enhancements") to anyone; however, if you choose to make your
Enhancements available either publicly, or directly to Lawrence Berkeley
National Laboratory, without imposing a separate written license agreement
for such Enhancements, then you hereby grant the following license: a
non-exclusive, royalty-free perpetual license to install, use, modify,
prepare derivative works, incorporate into other computer software,
distribute, and sublicense such enhancements or derivative works thereof,
in binary and source code form.
