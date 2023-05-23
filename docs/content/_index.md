<!--
This is a comment.
-->

{{< centerLinks "https://arxiv.org/pdf/2304.02532" "https://github.com/intuitive-robots/beso" "https://roboticsconference.org/" >}}



<img src="images/Beso_Figure_1.png" alt="drawing" width="100%">

We propose a new policy representation based on score-based diffusion models (SDMs).
We apply our new policy representation in the domain of Goal-Conditioned Imitation Learning (GCIL) to learn
general-purpose goal-specified policies from large uncurated datasets without rewards.
Our new goal-conditioned policy architecture "*BE*bhavior generation with *S*c*O*re-based Diffusion
Policies" (BESO) leverages a generative, score-based diffusion model as its policy.
BESO decouples the learning of the score model from the inference sampling process, and, hence
allows for fast sampling strategies to generate goal-specified behavior in just 3 denoising steps, compared to 30+ steps
of other diffusion based policies.
Furthermore, BESO is highly expressive and can effectively capture multi-modality present in the solution space of the
play data. Unlike previous methods such as Latent Plans or C-Bet, BESO does not rely on complex hierarchical policies or additional
clustering for effective goal-conditioned behavior learning. Finally, we show how BESO can even be used to learn a
goal-independent policy from play-data using classifier-free guidance. To the best of our knowledge this is the first
work that a) represents a behavior policy based on such a decoupled SDM b) learns an SDM based policy in the domain of
GCIL and c) provides a way to simultaneously learn a goal-dependent and a goal-independent policy from play-data.
We evaluate BESO through detailed simulation and show that it consistently outperforms several state-of-the-art
goal-conditioned imitation learning methods on challenging benchmarks.
We additionally provide extensive ablation studies and experiments to demonstrate the effectiveness of our method for goal-conditioned behavior generation.


### Experimental Results


We evaluate BESO on several challenging goal-conditioned imitation learning benchmarks and compare it to numerous state-of-the-art methods. We show that BESO consistently outperforms all baselines while only using 3 denoising steps. We additionally provide extensive ablation studies and experiments to demonstrate the effectiveness of our method for goal-conditioned behavior generation. See example rollouts below.

{{< doublevideo src1="images/kitchen_rollout.webm" src2="images/block_rollout.webm" title1="GC-Kitchen" title2="GC-Block Push" >}}

{{< doublevideo src1="images/beso_lh_seq.webm" src2="images/beso_single.webm" title1="CALVIN 2 Tasks" title2="CALVIN Hard Tasks" >}}

Check out our paper for detailed results and ablation studies.

### Classifier-free Guided Policy

Our experiments showcase the effectiveness Classifier-Free Guidance (CFG) Training of Diffusion Models in simultaneously learning goal-independent and goal-dependent policies. We can compose the gradients at test time to control the amount of goal-guidance we want to apply to the policy. The purpose of this setup is to demonstrate the influence of goal-guidance on the behavior of the policy. By gradually increasing the value of $\lambda$, we can observe how the policy becomes more goal-oriented and achieves a better success rate in accomplishing the desired goals.

Below you can see the performance of CFG-BESO on the kitchen and block push environment. When we set the guidance factor $\lambda=0$, we completely ignore the goal and generate random behavior with a high reward and low result (only gives credit, if a pre-defined goal is solved).

<img src="images/BESO_CFG_plot.png" alt="drawing" width="100%">

### BESO for General Imitation Learning

While BESO was initially designed for goal-conditioned imitation learning (IL), the general idea of using continuous-time diffusion models as a policy representation can be applied to standard (IL) and in hierarchical policies as well.
If you are interested in trying out BESO for Behavior Cloning, we build a BC-variant in a fork of the beautiful IL Benchmark Repo [Diffusion Policy](https://github.com/columbia-ai-robotics/diffusion_policy): [BESO Diffusion Policy](https://github.com/mbreuss/score_diffusion_policy). 

One of the key advantages of our BESO implementation is the use of a modular continuous-time diffusion model based on the work by [Karras et al. 2022](https://arxiv.org/abs/2206.00364). This modular approach allows for greater flexibility in adapting the sampler and adjusting the number of diffusion steps during inference, leading to improved performance. Additionally, our implementation enables fast action diffusion in just three steps.


## BibTeX

```bibtex
@inproceedings{
    reuss2023goal,
    title={Goal Conditioned Imitation Learning using Score-based Diffusion Policies},
    author={Reuss, Moritz and Li, Maximilian and Jia, Xiaogang and Lioutikov, Rudolf},
    booktitle={Robotics: Science and Systems},
    year={2023}
}
```

## Acknowledgements

The work presented here was funded by the German Research Foundation (DFG) – 448648559.
