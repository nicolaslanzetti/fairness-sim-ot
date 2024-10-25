---
title: "Social Influence Maximization using Optimal Transport"
author: "Shubham Chowdhary"
bibliography: references.bib
---

# Social Influence Maximization using Optimal Transport

<small>

**Authors:** Shubham Chowdhary, Giulia De Pasquale, Nicolas Lanzetti, Ana-Andreea Stoica, and Florian Dörfler

**Paper:** [Chowdhary et al., "Fairness in Social Influence Maximization via Optimal Transport", NeurIPS, 2024](https://arxiv.org/abs/2406.17736)

This repository containts the code for the research project "Fairness in Social Influence Maximization via Optimal Transport", presented at the NeurIPS 2024.

</small>

## <a id="env-setup"></a> Setting up the environment

### <a id="docker-install"></a> A. Via Docker
<small> *Instead, you can refer to [Section B](#conda-direct-install) below for a simpler architecture dependent installation below.* </small>
1. Make sure you have installed [Docker](https://docs.docker.com/engine/install/) for your device using appropriate steps.

2. Change your current working directory (termed `PROJECT_ROOT` hereby) to the root folder of this project. 

3. The project contains a pre-configured `Dockerfile` to generate appropriate **Linux** image for you and then install **miniconda** in it. **You do not have to explicitly install them on your machine**. Just build the Docker image and run it using the following commands,

```
# build the appropriate/supported Linux environment image (~3-5 mins, one time)
docker buildx build --platform=linux/amd64 --no-cache -t fairness-sim-ot-lab-img .

# run the built image on your machine
docker run --platform linux/amd64  -p 8888:8888 -v $(pwd):/app -it fairness-sim-ot-lab-img
```

Make sure to keep port **8888** free as we use it for running our Jupyter notebook on it. In case you want to change the port number, change it to an appropriate value in the `Dockerfile` as well as the *run command* above.

4. After this, a compatible Linux environment is installed on your local machine, and an appropriate miniconda package is used to create a *conda environemnt* supporting all the libraries you need to run this project. Your terminal should now prompt you with a **local URL to the hosted Jupyter Notebook**. Visit the same to start using the codebase! 

5. **[Optional]** Now you can place datasets into the `data/` folder as explained in [Project Structure::`data`](#project-struct-data) below, or utilize small ones that already come pre-installed with the project clone. You are now ready to walk thorugh the `main.ipynb` to get some idea about the library interfaces and then use the library functions for your own experiments!

<small>

**Note**: 
*(1) Any changes you make to the project files cloned on the disk are reflected on the docker environment-based project structure and the hosted Jupyter Notebook too. So you can make changes to the file structure as you wish. We use Docker's disk mounting feature to achieve this. In case it doesn't work for you just make sure that Docker has appropriate permissions to access and mount the disk where you cloned the project.*

*(2) Docker containerization is only meant to allow users/researchers to use the library for experimentation across all docker-supported platforms. Our library itself is purely a mathematical/statistical construct and doesn't dendeccarily need particular hardware requirements as set up through Docker.*

*(3) The Docker environemnt created is a **Linux/x86_64** one as our library was originally developed on it. If one can create an equivalent `conda` environment as in `environment.yml`, they can still use the library out of the box. To setup/start their custom environment they edit/use `lab_start.sh` directly without depending on Docker. Out of the box the script is also used by `Dockerfile` to do so.*

*(4) `Dockerfile` is configured to create an image that sets up a compatible environment and then run the Jupyter Notebook from the associated conda environment directly for ease of use. In case one prefers a direct shell access to the Docker environment and then proceed as they wish, they can edit the `Dockerfile` by commenting the `CMD` from it, and then build, run, and use the image as before.*
</small>

### <a id="conda-direct-install"></a> B. Via direct dependencies (more direct/easier setup)

We also list the minimal packages we need to setup the development environment for **any platform** in `minimal-environment.yml`. The environment created through this method is still *functionally* equivalent to what we need to use our library and is easier to directly work with if you have conda installed on your system which supports all the libraries listed in this environment configuration file. Here are the steps to create the environemnt directly on your system *(ideally works on Linux, Windows, and Mac (Intel, M-chip) architectures)*,

1. Ensure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed and configured for your system.

2. Change your current working directory (termed `PROJECT_ROOT` hereby) to the root folder of this project. 

3. Use `conda` and `minimal-environment.yml` to re-create the development environment for use as follows,

```
conda env create -n fairness-sim-ot-equiv-lab -f minimal-environment.yml -y
conda activate fairness-sim-ot-equiv-lab
```

4. **[Optional]** Now you can place datasets into the `data/` folder as explained in [Project Structure::`data`](#project-struct-data) below, or utilize small ones that already come pre-installed with the project clone. You are now ready to walk thorugh the `main.ipynb` to get some idea about the library interfaces and then use the library functions for your own experiments!


_Go on with the experiments!_


## <a id="project-struct"></a> Project Structure
We have created a **modularized pipeline** for streamlined experiments on **Social Influence Maximization** (SIM) using Random Graphs and Optimal Transport. The underlying theory is governed by *optimization in the probability distribution space* generated due to *exponentially many combinatorial actions*. 

The project strcuture is as follows,

```
adhoc_scripts/
data/
early_seed_selection/
main.ipynb
metrics/
optimize/
pipeline/
plots/
propagation/
utils/
```

<a id="project-struct-data"></a> 1. `data/`
It contains separate folders (named the same as the dataset label used to identify them in the code everywhere) for each of the datasets used. **Some folders (eg. for Instagram, Deezer, DBLP) are expected to be empty due to large dataset upload restrictions on git remote servers.**

All the datasets are normalized into a uniform structure for its easy interaction with the rest of the code. For this, look into the `<data-set-label>/raw/` folder for all the open-source "mis-structured" files/folders available for the dataset. **If this folder is unavailable/empty, this is what one creates and put open-source dataset "as is" in.** We have created `<data-set-label>/normalize_ds.py` to generate `<data-set-label>/graph_edges.csv`, `<data-set-label>/graph_node_features.csv`, and `<data-set-label>/metadata.txt` to be further usable uniformly by the code by processing the raw dataset files.

For easy creation of custom `normalize_ds.py` we have some commonly used utils/routines available in `utils/data_utils.py`.

2. `early_seed_selection/`
All the algorithms for initial seedset selection go here in a separate folder. Our baselines, which are essentailly heuristics for fair seed selection, are implemented here.

3. `metrics/`
Implements several metrics used to quantify the progress/results of seedset selection and propagation. Each type of metric (describing groups reached, nodes covered, seed selection, or Optimal Transport related) gets a different file.

4. `optimize/`
Implements the **Stochastic Seedset Selection Descent** (S3D) to iteratively optimize seedset selection for the best $\beta$-fairness

5. `pipeline/`
Integrates all the modules into a single interface for quick experiemnts based on config options

6. `plots/`
Code for all type of plots (joint distribution, E-f plots, etc) go here.

7. `propagation/`
Different information propagation algorithms come here. `propagate.py` implements the widely used **Independent Cascade** for random graphs. Other implementations can we added as separate files. `multiple_propagate` performs the given `prop_routine` several times and aggregates them for monte-carlo inferences.

8. `utils/`
Common one-stop for all the generic utils for repeated computation, managing logger, OT matrices, dataset preprocessing, etc.

9. `adhoc_scripts/`
Python script **not** a part of the main library but is useful for converting data formats (tikz, csv), or running custom experiments as a monolithic process on remote compute nodes. Modify to run/reuse them as and when usecase overlaps. 

10. `main.ipynb`
The **main introduction and interface** script to walk new users through the library interface.

A more granular documented code structure is hosted at [Read the Docs](https://fairness-sim-ot.readthedocs.io/en/latest/) tied to the public [repository](https://github.com/nicolaslanzetti/fairness-sim-ot).

## <a id="citation"></a> Citation

If you use this code in an academic context, please cite:

```bibtex
@inproceedings{chowdhary2024fairness,
  title={Fairness in Social Influence Maximization via Optimal Transport},
  author={Chowdhary, Shubham and De Pasquale, Giulia and Lanzetti, Nicolas and Stoica, Ana-Andreea and D{\"o}rfler, Florian},
  booktitle={NeurIPS},
  year={2024},
  organization={PMLR}
}
```

## <a id="contrib"></a> Contribution
_We hope the open-source community adapts this framework and continues to study and contribute in this line of research._

Thanks,\
_Authors_
