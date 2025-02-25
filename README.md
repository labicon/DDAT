# DDAT: Diffusion policies enforcing Dynamically Admissible robot Trajectories

In this project we create a novel diffusion architecture to generate dynamically feasible robot trajectories by incorporating autoregressive projections in the training and inference phase of a diffusion transformer.

[Paper](https://arxiv.org/pdf/2502.15043.pdf) [Website](https://iconlab.negarmehr.com/DDAT/)

![DDAT training loop](docs/pictures/DDAT_scheme.svg)


## Overview

Diffusion models are stochastic by nature.
Thus, the trajectories they generate **cannot** satisfy exactly the equations of motions of robots.
When deploying such *infeasible* trajectories, the actual robot diverges from the prediction and most likely fails to accomplish its task.
Previous works have thus focused on replanning the entire trajectory very frequently.
We propose to address the root cause of the problem by forcing our diffusion models to generate **fesible** trajectories.

![infeasible trajectories diverge](docs/videos/GO2_smaller_gif.gif)


## Theory

We assume to have a black-box discrete-time simulator $f$ of a robot state 

$$s_{t+1} = f(s_t, a_t)$$

controlled by actions $a_t \in \mathcal A$.
A trajectory $\tau = \\{s_0, s_1, ...\\}$ is **admissible** if each $s_{t+1}$ is reachable from its predecessor $s_t$ with an action $a_t \in \mathcal{A}$. In other words $s_{t+1}$ must be in the **reachable set** $\mathcal{R}(s_t)$ of its predecessor $s_t$ where

$$\mathcal{R}(s_t) = \\{ f(s_t, a)\ \text{for all}\ a \in \mathcal{A} \\}.$$

To generate admissible trajectories we then design autoregressive projectors $\mathcal P$ iteratively projecting each $s_{t+1}$ onto $\mathcal{R}(s_t)$ for all $t$.

## Organization

- `docs` : all the elements to build the [project website](https://iconlab.negarmehr.com/DDAT/)
- `code` : our implementation of DDAT with diffusion transformers, projectors, and trained models.


## Citation
```
@inproceedings{bouvier2025ddat,
        title = {DDAT: Diffusion Policies Enforcing Dynamically Admissible Robot Trajectories},
        author = {Bouvier, Jean-Baptiste and Ryu, Kanghyun and Nagpal, Kartik and Liao, Qiayuan and Sreenath, Koushil and Mehr, Negar},
        booktitle = {arxiv.org/abs/2502.15043},
        year = {2025}
      }
```

## Acknowledgments

Our diffusion transformer architecture is largely based on the [AlignDiff code](https://github.com/ZibinDong/AlignDiff-ICLR2024/tree/main).
