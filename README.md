# MMVL -  Multi-Map Visual Localization

> [!WARNING]
> This is a work in progress. We are missing two necessary models for running the solution.

## Requirements
- Docker installed


## Setup
0. Download the dataset ([original one available here](https://zenodo.org/records/13748357)) and extract all flights into a `dataset` folder. It should look like:
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
- src/
```

1. Build the container by running
```bash
docker build -t mmvl:latest .
```

2. Run the Docker container in interactive mode
```bash
docker container run -it mmvl bash
```

3. Run the application
```bash
./main <path to dataset> <number of runs> <maps>
```

For instance, if running `Flight4`, the command is:
```bash
./main ../dataset/Flight4 2 UFRGS-01-2017
```

### Notas
Modelos necessários:
 - aug4Run_resnet34_30norm_siam.pt
 - aug4Run_resnet34_30norm_combined.pt