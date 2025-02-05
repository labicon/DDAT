# DDAT: Diffusion policies enforcing Dynamically Admissible robot Trajectories

In this project we create a novel diffusion architecture to generate dynamically admissible robot trajectories

## Theory

Given a black-box discrete-time simulator $f$ of a robot state 
$$s_{t+1} = f(s_t, a_t)$$
controlled by actions $a_t \in \mathcal A$.
A trajectory $\tau = \\{s_0, s_1, ...\\}$ is <em>admissible</em> if each $s_{t+1}$ is reachable from its predecessor $s_t$ with an action $a_t \in \mathcal{A}$. In other words $s_{t+1}$ must be in the reachable $\mathcal{R}(s_t)$ set of its predecessor $s_t$ where
$$\mathcal{R}(s_t) := \\big\{ f(s_t, a)\ \text{for all}\ a \in \mathcal{A} \\big\}.$$


## Code

