LingBot-VA 在线 RL（Flow-SDE / Flow-Noise GRPO）
================================================

本文介绍如何在 LIBERO-Object 上，使用 RLinf 自带的流匹配（flow-matching）RL
算法 **Flow-SDE** 与 **Flow-Noise**，对 **LingBot-VA** 进行在线 RL 微调。该实现
建立在 LingBot-VA 的 SFT + 评测集成之上，并复用 RLinf 已为 ``openpi``
（:math:`\pi_0` / :math:`\pi_{0.5}`）和 ``lingbotvla`` 提供的流匹配 RL 组件，
将其适配到 LingBot-VA（Wan 2.2 视频扩散 Transformer）的 **动作流** 上。


动机
----

LingBot-VA 通过流匹配 ODE 逐步去噪动作隐变量来生成动作（评测后端中的
``action_scheduler.step`` 欧拉更新）。确定性 ODE 的逐动作对数似然不可解，
因此无法直接用策略梯度 RL 优化。RLinf 提供两种解法（见 ``π_RL``，
arXiv:2510.25889；以及 Flow-GRPO，arXiv:2505.05470）：

- **Flow-SDE**：将去噪 ODE 转换为等价 SDE，使每个去噪步成为带闭式对数概率的
  高斯转移，SDE 扩散项提供探索噪声，**不引入新参数**。
- **Flow-Noise**：保持 ODE 均值，额外引入一个**可学习**的高斯噪声网络，将去噪
  建模为离散时间 MDP，对数概率精确可解。

两者都把多步去噪器变成一串高斯转移，从而可被 GRPO 优化，与 ``lingbotvla`` 一致。


在 LingBot-VA 上的映射
----------------------

Wan 的 ``FlowMatchScheduler`` 将前向过程参数化为
``x_sigma = (1 - sigma) * x0 + sigma * x1``（``x1 ~ N(0, I)``），并预测速度
``v = x1 - x0``。这与 RLinf 其他动作专家的约定一致，只是把连续时间 ``t`` 换成了
调度器的 ``sigma``。逐步高斯转移
（``rlinf/models/embodiment/lingbotva/flow_rl.py``）因此数学上完全相同::

    x0_pred = x_t - sigma * v
    x1_pred = x_t + (1 - sigma) * v

    # Flow-SDE
    sigma_i  = noise_level * sqrt(sigma / (1 - sigma))
    mean     = x0_pred * (1 - sigma_next)
             + x1_pred * (sigma_next - sigma_i**2 * delta / (2 * sigma))
    std      = sqrt(delta) * sigma_i

    # Flow-Noise
    mean     = x0_pred * (1 - sigma_next) + x1_pred * sigma_next
    std      = noise_head(sigma)          # 可学习

遵循 Flow-GRPO，每条 rollout 只在**一个**去噪步上随机采样（其余为确定性 ODE
步），且只有该步的对数概率进入策略梯度；``joint_logprob: True`` 则切换为对所有
步求平均。

两条执行路径
~~~~~~~~~~~~

与 ``lingbotvla``（单一 ``nn.Module``）不同，LingBot-VA 有两条彼此独立、由权重
同步器保持一致的 Transformer 路径：

- **Rollout worker**（``lingbotva.training_mode=False``）：通过评测后端
  （``VA_Server``）采样动作。``predict_action_batch(mode="train")`` 先确定性地
  去噪视频以预热 KV cache，再随机去噪动作流，返回真实的 ``prev_logprobs`` 以及
  重算所需的 ``chains`` 与条件张量。
- **Actor worker**（``lingbotva.training_mode=True``）：将 Transformer 暴露给
  FSDP。``default_forward`` 在保存的 ``chains`` 上带梯度地重算速度（及可学习噪声
  标准差），返回 GRPO 的 ``logprobs`` / ``values`` / ``entropy``。


配置
----

推荐配置：

- ``examples/embodiment/config/libero_object_grpo_lingbotva.yaml``

主要开关（位于 ``actor.model`` 下）：

.. code:: yaml

   noise_method: flow_sde   # flow_sde | flow_noise
   noise_level: 0.7         # SDE 扩散系数
   joint_logprob: False     # 单步（默认）或全步对数概率
   add_value_head: False    # GRPO 为 critic-free
   lingbotva:
     training_mode: True               # actor：FSDP Transformer
     action_num_inference_steps: 10    # 动作去噪步数

通过一条命令行覆盖切换到 Flow-Noise::

   bash examples/embodiment/run_embodiment.sh libero_object_grpo_lingbotva \
       actor.model.noise_method=flow_noise rollout.model.noise_method=flow_noise \
       algorithm.entropy_bonus=0.01

rollout 模型的 ``noise_method`` / ``noise_level`` 与 actor 保持一致，以保证采集的
``prev_logprobs`` 与重算的对数概率一致（GRPO 假设采集时刻重要性比为 1）。


验证状态
--------

- **流匹配 RL 核心数学**：在 ``tests/unit_tests/test_lingbotva_flow_rl.py`` 中有
  单元测试（Flow-SDE / Flow-Noise / Flow-ODE 转移、对数概率与
  ``torch.distributions`` 对齐、``noise_level → 0`` 的 ODE 极限、可学习噪声头，
  以及最关键的 rollout↔重算对数概率一致性，含逐样本索引）。这些可在纯 CPU 的
  ``torch`` 环境下运行。
- **Wan Transformer 接线**：rollout/重算的 Transformer 调用与 actor 侧 KV cache
  预热，与评测后端的确定性循环一致，但需在 GPU 上配合真实的 ``wan_va`` 包与 SFT
  权重做端到端验证（已在 ``LingbotVAActionModel._warm_video_cache`` 中标注）。
