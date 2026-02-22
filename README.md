![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=3776ab)
![Docker 4.55.0](https://img.shields.io/badge/docker-4.55.0-2496ed?logo=docker&logoColor=2496ed)
![PyTorch 1.13.1](https://img.shields.io/badge/pytorch-1.13.1-ee4c2c?logo=pytorch&logoColor=ee4c2c) 

# MMVL -  Multi-Map Visual Localization

This is a fork from [Tobias Lomo's MMVL](https://github.com/tobiaslo/MMVL), designed to be run on a docker container.


## Requirements
- [Docker Desktop](https://docs.docker.com/desktop/) installedssss
- At least 16 GB RAM

> [!NOTE]
> Solution tested on Windows 11.


## Setup
1. Download the dataset ([original one available here](https://zenodo.org/records/13748357)) and extract all flights into a `dataset` folder.

2. Download the trained models from the original repo releases, and save them in the `models` folder: [siam model](https://github.com/tobiaslo/MMVL/releases/download/v1.0-models/aug4Run_resnet34_30norm_siam.pt) and [combined model](https://github.com/tobiaslo/MMVL/releases/download/v1.0-models/aug4Run_resnet34_30norm_combined.pt).

The final structure should look like:
It should look like:
```bash
mmvl
- dataset/
--- Flight1/
------ Flight/
------ Maps/
------ odom.txt
------ traj.txt
--- Flight2/
...
- models/
--- aug4Run_resnet34_30norm_combined.pt
--- aug4Run_resnet34_30norm_siam.pt
- src/
--- *.h
--- *.cpp
```

3. Build the container by running
```bash
docker build -t mmvl:latest .
```

4. Run the Docker container in interactive mode
```bash
docker container run -it mmvl bash
```

5. In the interactive command line, build the app
```bash
mkdir build && cd build && \
    cmake .. && \
    cmake --build . --config Release

cd app/build
```

6. Run the application
```bash
./main <path to dataset> <number of runs> <maps>
```

For instance, if running `Flight4`, the command is:
```bash
./main ../dataset/Flight4 2 UFRGS-01-2017
```

7. The results will be stored in the `app/build` folder, under the name `result.txt`. To copy it to your host, use:
```bash
docker cp <container_id>:/app/build/result.txt .\your-destination-folder
```

### Notas
Modelos necessários:
 - aug4Run_resnet34_30norm_siam.pt
 - aug4Run_resnet34_30norm_combined.pt