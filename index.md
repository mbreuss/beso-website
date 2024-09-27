---
layout: project_page
permalink: /

title: "Goal Conditioned Imitation Learning using Score-based Diffusion Policies"
authors: <a href="https://mbreuss.github.io/">Moritz Reuss</a>, Maximilian Xiling Li, Xiaogang Jia, <a href="https://rudolf.intuitive-robots.net/">Rudolf Lioutikov</a>
affiliations: <a href="https://www.irl.iar.kit.edu/">KIT Intuitive Robots Lab</a>
venue: "RSS 2023"
paper: https://arxiv.org/pdf/2304.02532
# video:
code: https://github.com/intuitive-robots/beso
# data:
full-page-landing-include: fpl-video.html
---


---

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
We propose a new policy representation based on score-based diffusion models (SDMs). We apply our new policy representation in the domain of Goal-Conditioned Imitation Learning (GCIL) to learn general-purpose goal-specified policies from large uncurated datasets without rewards. Our new goal-conditioned policy architecture "<b>BE</b>havior generation with <b>S</b>c<b>O</b>re-based Diffusion Policies" (BESO) leverages a generative, score-based diffusion model as its policy. BESO decouples the learning of the score model from the inference sampling process, and, hence allows for fast sampling strategies to generate goal-specified behavior in just 3 denoising steps, compared to 30+ steps of other diffusion based policies. Furthermore, BESO is highly expressive and can effectively capture multi-modality present in the solution space of the play data. Unlike previous methods such as Latent Plans or C-Bet, BESO does not rely on complex hierarchical policies or additional clustering for effective goal-conditioned behavior learning. Finally, we show how BESO can even be used to learn a goal-independent policy from play-data using classifier-free guidance. To the best of our knowledge this is the first work that a) represents a behavior policy based on such a decoupled SDM b) learns an SDM based policy in the domain of GCIL and c) provides a way to simultaneously learn a goal-dependent and a goal-independent policy from play-data. We evaluate BESO through detailed simulation and show that it consistently outperforms several state-of-the-art goal-conditioned imitation learning methods on challenging benchmarks. We additionally provide extensive ablation studies and experiments to demonstrate the effectiveness of our method for goal-conditioned behavior generation.
        </div>
    </div>
</div>

<div class="columns is-centered">
    <img src="/static/image/Beso_Figure_1.png" alt="BESO" class="column is-four-fifths">
</div>

---

## Experimental Results
We evaluate BESO on several challenging goal-conditioned imitation learning benchmarks and compare it to numerous state-of-the-art methods. We show that BESO consistently outperforms all baselines while only using 3 denoising steps. We additionally provide extensive ablation studies and experiments to demonstrate the effectiveness of our method for goal-conditioned behavior generation. See example rollouts below.

<div class="columns is-full is-centered has-text-centered">
    <div class="column is-four-fifths">
        <div class="column is-full columns is-centered">
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/kitchen_rollout.webm">
                </video>
                <p>GC-Kitchen</p>
            </div>
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/block_rollout.webm">
                </video>
                <p>GC-Block Push</p>
            </div>
        </div>
        <div class="column is-full columns is-centered">
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/beso_lh_seq.webm">
                </video>
                <p>CALVIN 2 Tasks</p>
            </div>
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/beso_single.webm">
                </video>
                <p>CALVIN Hard Tasks</p>
            </div>
        </div>
        <div class="column is-full columns is-centered">
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/rp_inside.mkv">
                </video>
                <p>D3IL Align (inside)</p>
            </div>
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/rp_outside.mkv">
                </video>
                <p>D3IL Align (outside)</p>
            </div>
        </div>
    </div>
</div>

### Real World Experiments
We evaluate BESO on a challenging real world toy kitchen environment with 10 tasks and compare it to the BC baseline.
<div class="columns is-full is-centered has-text-centered">
    <div class="column is-four-fifths">
        <div class="column is-full columns is-centered">
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/real_robot_50/banana_from_sink_to_right_stove_7.mp4">
                </video>
            </div>
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/bc_real_robot_50/banana_from_sink_to_right_stove_37.mp4">
                </video>
            </div>
        </div>
        <div class="column is-full columns is-centered">
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/real_robot_50/pickup_toast_and_put_to_sink_29.mp4">
                </video>
            </div>
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/bc_real_robot_50/pickup_toast_and_put_to_sink_45.mp4">
                </video>
            </div>
        </div>
        <div class="column is-full columns is-centered">
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/real_robot_50/open_oven_22.mp4">
                </video>
            </div>
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/bc_real_robot_50/open_oven_11.mp4">
                </video>
            </div>
        </div>
        <div class="column is-full columns is-centered">
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/real_robot_50/pot_from_sink_to_right_stove_31.mp4">
                </video>
            </div>
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/bc_real_robot_50/pot_from_sink_to_right_stove_7.mp4">
                </video>
            </div>
        </div>
        <div class="column is-full columns is-centered">
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/real_robot_50/pull_oven_tray_13.mp4">
                </video>
                <p>BESO</p>
            </div>
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/bc_real_robot_50/pull_oven_tray_17.mp4">
                </video>
                <p>BC</p>
            </div>
        </div>
    </div>
</div>

#### Failure Cases in Real Robot Experiments
<div class="columns is-full is-centered has-text-centered">
    <div class="column is-four-fifths">
        <div class="column is-full columns is-centered">
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/real_robot_50/pot_from_sink_to_right_stove_11.mp4">
                </video>
                <p>Task: Move the pot from the sink to the right stove. BESO went to the stove instead of the sink, and failed to pickup the pot afterwards.</p>
            </div>
            <div class="column is-half">
                <video width="100%" autoplay muted loop playsinline>
                    <source src="/static/video/real_robot_50/pickup_toast_and_put_to_sink_49.mp4">
                </video>
                <p>Task: Pick up the toast and put it into the sink. BESO didn't lift the toast high enough and carried the toaster too.</p>
            </div>
        </div>
    </div>
</div>

## Classifier-Free Guided Policy

Our experiments showcase the effectiveness Classifier-Free Guidance (CFG) Training of Diffusion Models in simultaneously learning goal-independent and goal-dependent policies. We can compose the gradients at test time to control the amount of goal-guidance we want to apply to the policy. The purpose of this setup is to demonstrate the influence of goal-guidance on the behavior of the policy. By gradually increasing the value of lamda, we can observe how the policy becomes more goal-oriented and achieves a better success rate in accomplishing the desired goals.

Below you can see the performance of CFG-BESO on the kitchen and block push environment. When we set the guidance factor lamda=0, we completely ignore the goal and generate random behavior with a high reward and low result (only gives credit, if a pre-defined goal is solved).

<div class="columns is-centered">
    <img src="/static/image/BESO_CFG_plot.png" alt="BESO CFG" class="column is-four-fifths">
</div>

## Using BESO in your own project

While BESO was initially designed for goal-conditioned imitation learning (IL), the general idea of using continuous-time diffusion models as a policy representation can be applied to standard (IL) and in hierarchical policies as well.
If you are interested in trying out BESO for Behavior Cloning, we build a BC-variant in a fork of the beautiful IL Benchmark Repo: [BESO Diffusion Policy](https://github.com/mbreuss/score_diffusion_policy).

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

## Related Projects
<h3><a href="https://intuitive-robots.github.io/mdt_policy/">Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals</a></h3>
<div class="column is-full columns">
    <div class="column is-half">
        <img src="/static/image/mdt-v-figure.png" alt="MDT-V Overview">
    </div>
    <div class="column is-half">
        <p>
        The Multimodal Diffusion Transformer (MDT) is a novel framework that learns versatile behaviors from multimodal goals with minimal language annotations. Leveraging a transformer backbone, MDT aligns image and language-based goal embeddings through two self-supervised objectives, enabling it to tackle long-horizon manipulation tasks. In benchmark tests like CALVIN and LIBERO, MDT outperforms prior methods by 15% while using fewer parameters. Its effectiveness is demonstrated in both simulated and real-world environments, highlighting its potential in settings with sparse language data.
        </p>
    </div>
</div>

<h3><a href="https://robottasklabeling.github.io/">Scaling Robot Policy Learning via Zero-Shot Labeling with Foundation Models</a></h3>
<div class="column is-full columns">
    <div class="column is-half">
        <img src="/static/image/nils-ow.png" alt="NILS Overview">
    </div>
    <div class="column is-half">
        <p>
Using pre-trained vision-language models, NILS detects objects, identifies changes, segments tasks, and annotates behavior datasets. Evaluations on the BridgeV2 and kitchen play datasets demonstrate its effectiveness in annotating diverse, unstructured robot demonstrations while addressing the limitations of traditional human labeling methods.
        </p>
    </div>
</div>