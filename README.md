# Deep Bayesian Active Learning for Accelerating Stochastic Simulation

## Paper: 
Dongxia Wu, Ruijia Niu, Matteo Chinazzi, Alessandro Vespignani, Yi-An Ma, Rose Yu, [Deep Bayesian Active Learning for Accelerating Stochastic Simulation](https://arxiv.org/pdf/2106.02770.pdf), KDD 2023

## Requirements

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Abstract
Stochastic simulations such as large-scale, spatiotemporal, age-structured epidemic models are computationally expensive at fine-grained resolution. While deep surrogate models can speed up the simulations, doing so for stochastic simulations and with active learning approaches is an underexplored area. We propose Interactive Neural Process (INP), a deep Bayesian active learning framework for learning deep surrogate models to accelerate stochastic simulations. INP consists of two components, a spatiotemporal surrogate model built upon Neural Process (NP) family and an acquisition function for active learning. For surrogate modeling, we develop Spatiotemporal Neural Process (STNP) to mimic the simulator dynamics. For active learning, we propose a novel acquisition function, Latent Information Gain (LIG), calculated in the latent space of NP based models. We perform a theoretical analysis and demonstrate that LIG reduces sample complexity compared with random sampling in high dimensions. We also conduct empirical studies on three complex spatiotemporal simulators for reaction diffusion, heat flow, and infectious disease. The results demonstrate that STNP outperforms the baselines in the offline learning setting and LIG achieves the state-of-the-art for Bayesian active learning. 

## Description
1. seir/: INP on SEIR simulator for active learning.
2. reaction_diffusion/: INP on reaction diffusion simulator for active learning. STNP on reaction diffusion simulator for surrogate modeling.
3. heat/: INP on heat simulator for active learning. STNP on heat simulator for surrogate modeling.
4. leam_us/: INP on LEAM-US simulator for active learning. STNP on LEAM-US simulator for surrogate modeling.

## Dataset Download

### Reaction Diffusion
```
cd reaction_diffusion/
wget -O data.zip https://roselab1.ucsd.edu/seafile/f/f6145ace6c984256904f/?dl=1
unzip data.zip
```

### Heat
```
cd heat/
wget -O data.zip https://roselab1.ucsd.edu/seafile/f/1457be6debeb484fb72c/?dl=1
unzip data.zip
```

### LEAM-US
```
cd leam_us/active/data
wget -O data.zip https://roselab1.ucsd.edu/seafile/f/766ef6512667486b8b2c/?dl=1
unzip data.zip
cp -r data ../../offline/data
```


## Model Training and Evaluation


### SEIR
```
cd seir/
python seir.py
```

### Reaction Diffusion (Active Learning)
```
cd reaction_diffusion/active
python rd_active_lig.py
```

### Reaction Diffusion (Surrogate Modeling)
```
cd reaction_diffusion/offline
python rd_offline_stnp.py
```

### Heat (Active Learning)
```
cd heat/active
python heat_active_lig.py
```

### Heat (Surrogate Modeling)
```
cd heat/offline
python heat_offline_stnp.py
```

### LEAM-US (Active Learning)
```
cd leam_us/active
python leam_us_active_lig.py --config_filename=data/model/dcrnn_cov.yaml
```

### LEAM-US (Surrogate Modeling)
```
cd leam_us/offline
python leam_us_offline_stnp.py --config_filename=data/model/dcrnn_cov.yaml
```



## Cite
```
@inproceedings{wu2023deep,
  title={Deep Bayesian Active Learning for Accelerating Stochastic Simulation},
  author={Wu, Dongxia and Niu, Ruijia and Chinazzi, Matteo and Vespignani, Alessandro and Ma, Yi-An and Yu, Rose},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  year={2023}
}
```