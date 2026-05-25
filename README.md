# OPSD：On-Policy Self-Distillation

本目录收录了 OPSD（On-Policy Self-Distillation）相关的可直接运行配方：

- [`run_qwen_opsd.sh`](./run_qwen_opsd.sh)：原始 OPSD 论文复现
- [`run_qwen_sdpo.sh`](./run_qwen_sdpo.sh)：SDPO（EMA teacher + alpha-KL）
- [`run_qwen_vision_opd.sh`](./run_qwen_vision_opd.sh)：Vision-OPD（多模态图像 prompt 改写）
- [`teacher_dataloader/`](./teacher_dataloader/)：上述三个配方对应的 teacher dataloader 实现
- [`vision_opd_reward.py`](./vision_opd_reward.py)：Vision-OPD 用的 V\*Bench multi-choice reward function

OPSD 让 student **以自身为教师**进行 on-policy 蒸馏——不需要额外的教师模型、不需要额外的显存。教师信号通过**重写 prompt**（注入 gold answer、加 few-shot 示范、加 hint 等）从同一个模型激发出来。

> 适用条件：没有更强的教师模型，但你能写出"如果给模型多一点信息，它能答得更好"的 prompt 改写规则。

## 环境安装
参考 `install_for_cuda128.sh` 和 `install_for_cuda130.sh`

---

## 1. 快速上手

### Step 1：写一个 teacher dataloader

继承 `OfflineTeacherDataloader`（无需 rollout 信息）或 `OnlineTeacherDataloader`（需要访问 student rollout），实现 `build_one` 返回 `TeacherSample`：

#### 示例 1：Offline —— 把 gold answer 拼进 prompt template

```python
# my_pkg/opsd.py
import copy
from verl.workers.self_distillation import OfflineTeacherDataloader, TeacherSample


class GoldAnswerTeacher(OfflineTeacherDataloader):
    """把 gold answer 嵌入用户消息末尾，引导 ref policy 生成更对齐的 logprob。"""

    HINT_SUFFIX = "\n\n(Hint: the final answer is {answer}. Show your reasoning step by step.)"

    def build_one(self, *, prompt_messages, prompt_text, multi_modal_data, extra_info):
        answer = extra_info.get("answer")
        if answer is None or prompt_messages is None:
            return TeacherSample(messages=[], skip=True)

        messages = copy.deepcopy(prompt_messages)
        # 在最后一条 user 消息末尾追加 hint
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["content"] += self.HINT_SUFFIX.format(answer=answer)
                break
        return TeacherSample(messages=messages, multi_modal_data=multi_modal_data)
```

#### 示例 2：Online —— 用同 prompt 下最优兄弟 response 当 few-shot

```python
# my_pkg/opsd.py
import copy
from verl.workers.self_distillation import OnlineTeacherDataloader, TeacherSample


class BestSiblingTeacher(OnlineTeacherDataloader):
    """用同一 prompt 下奖励最高的兄弟 response 当 few-shot 示范（需 rollout.n > 1）。"""

    def build_one(self, *, prompt_messages, prompt_text, response_text,
                  reward, multi_modal_data, extra_info, batch_view, index):
        if prompt_messages is None:
            return TeacherSample(messages=[], skip=True)

        best_j, best_r = index, reward if reward is not None else -float("inf")
        for j in batch_view.iter_same_uid(index):
            rj = batch_view.rewards[j] if batch_view.rewards else None
            if rj is not None and rj > best_r:
                best_j, best_r = j, rj
        if best_j == index:
            return TeacherSample(messages=[], skip=True)

        demo = batch_view.response_texts[best_j]
        messages = [
            *copy.deepcopy(prompt_messages),
            {"role": "assistant", "content": demo},
            {"role": "user", "content": "Now answer the same question:"},
        ]
        return TeacherSample(messages=messages, multi_modal_data=multi_modal_data)
```

#### `build_one` 入参

| 参数 | 说明 |
| --- | --- |
| `prompt_messages` | 原始 chat messages（含多模态 segments） |
| `multi_modal_data` | dict，key 为 `images` / `videos` / `audios` |
| `extra_info` | 数据集行的 `extra_info`（gold answer、data_source 等） |
| `prompt_text` | prompt 文本形式（调试用） |

`OnlineTeacherDataloader` 额外接收 `response_text` / `reward` / `batch_view` / `index`。

无效样本返回 `TeacherSample(messages=[], skip=True)`，该条蒸馏 loss 会被自动 mask。

**注意事项**：
- 修改 `prompt_messages` 前一定 `copy.deepcopy`，否则会污染 batch 其他字段。
- `multi_modal_data` 使用**复数** key（`images` / `videos` / `audios`）。
- `__init__(**kwargs)` 接收 yaml 中 `self_distill.dataloader_kwargs` 的所有键值，用 `self.config["my_param"]` 访问。

### Step 2：启动训练

```bash
distillation.enabled=True \
distillation.mode=self \
distillation.self_distill.dataloader=my_pkg.opsd:GoldAnswerTeacher \
distillation.distillation_loss.loss_mode=k3 \
distillation.distillation_loss.use_task_rewards=True \
distillation.distillation_loss.use_policy_gradient=False
```

就这三条加上你已有的 PPO/GRPO 启动命令即可。**不需要**配 `distillation.teacher_models / n_gpus_per_node / nnodes`——OPSD 完全复用 colocated 的 ref policy。

完整脚本（通用 OPSD 模板）：[`../on_policy_distillation_trainer/run_qwen3_8b_opsd_fsdp.sh`](../on_policy_distillation_trainer/run_qwen3_8b_opsd_fsdp.sh)。

如果你想直接跑仓库里已经对齐过当前实现的配方，优先看本目录下的三个脚本：[`run_qwen_opsd.sh`](./run_qwen_opsd.sh) / [`run_qwen_sdpo.sh`](./run_qwen_sdpo.sh) / [`run_qwen_vision_opd.sh`](./run_qwen_vision_opd.sh)。

### Step 3：三个具体例子

下面三个示例展示了同一套 OPSD 框架如何覆盖 OPSD / SDPO / Vision-OPD。对应的可直接运行的 teacher dataloader 示例放在 [`./teacher_dataloader/`](./teacher_dataloader/)。

| 配方 | 启动脚本 | teacher 注入 | loss | teacher 更新 |
|---|---|---|---|---|
| **OPSD**（原 paper 复现） | [run_qwen_opsd.sh](./run_qwen_opsd.sh) | 把 `extra_info.{solution,reference_solution,answer}` 中的参考解答重写进最后一条 user turn | `kl_family=sdpo, sdpo.alpha=0.0, sdpo.mode=topk, sdpo.tail=renorm, topk=128, loss_max_clamp=0.05` | `ref` 固定 |
| **SDPO** | [run_qwen_sdpo.sh](./run_qwen_sdpo.sh) | 把同一 UID 下 reward 足够高的 sibling rollout 作为 reprompt 示范拼回 prompt | `kl_family=sdpo, sdpo.alpha=0.5, sdpo.mode=topk, sdpo.tail=add, sdpo.ratio_clip=2.0` | `ema`, `teacher_update_rate=0.05` |
| **Vision-OPD** | [run_qwen_vision_opd.sh](./run_qwen_vision_opd.sh) | teacher 侧复用原始文本 prompt，但把图像替换成 `extra_info.bbox_images` 指向的 bbox 版本 | `kl_family=sdpo, sdpo.alpha=0.5, sdpo.mode=topk, sdpo.tail=add, sdpo.ratio_clip=2.0` | `ema`, `teacher_update_rate=0.05` |

### Step 4：常用组合速查

- **基线（监督式 KD）**：`loss_mode=k3, use_policy_gradient=False` ——稳，先用这个跑通。
- **配合 GRPO**：`algorithm.adv_estimator=grpo, rollout.n=4`，蒸馏与 advantage 无耦合，正交叠加即可。
- **top-K 反向 KL**：`kl_family=sdpo, sdpo.alpha=1.0, sdpo.tail=add` ——更细致的分布级监督，慢一点。

---

## 2. 工作原理（30 秒理解）

```
student rollout (prompt, response)
    │
    ▼
your dataloader.build_one()  →  TeacherSample(messages=...)
    │
    ▼
verl 自动 tokenize teacher messages，拼上 student response tokens
    │
    ▼
ref policy forward → teacher_logprobs
    │
    ▼
复用 OPD loss kernel：蒸馏 loss 与任务奖励并行生效
```

关键点：
- Student 和 teacher 共用同一个进程、同一份 FSDP shard、同一个 tokenizer/processor。
- "教师"的本质只是**一次 prompt 不同的 ref forward**，因此 OPSD 没有任何额外 GPU 开销。
- 教师权重可以静态固定（`ref`），也可以让它跟着 student 走（`ema`/`progressive`/`trust_region`）。

---

## 3. 超参数参考

下面只列 OPSD（`distillation.mode=self`）下实际生效的字段，按命名空间分组。每条标注：类型、可选值/取值范围、默认值与作用。

### 3.1 顶层 `distillation.*`

- **`enabled`**（bool，`true` / `false`，默认 `false`）：蒸馏总开关，关闭时完全不走蒸馏分支。
- **`mode`**（str，`external` / `self`，默认 `external`）：教师来源。`external` 走外挂教师模型；`self` 复用同进程 ref policy，OPSD 必须设为 `self`。

### 3.2 蒸馏 loss `distillation.distillation_loss.*`

公共字段：

- **`kl_family`**（str，`verl` / `sdpo`，默认 `verl`）：loss 实现家族。`verl` 走原版 estimator/top-K kernel；`sdpo` 走 alpha-KL 路径，并按 `sdpo.mode` 自动改写 `loss_mode`。
- **`loss_mode`**（str，默认 `k3`）：具体的 loss kernel，可选：
  - 单样本 KL 估计器：`k1`、`k2`、`k3`、`kl`、`low_var_kl`；
  - logprob 距离：`abs`（绝对值）、`mse`（平方）；
  - top-K 分布级 KL：`forward_kl_topk`、`reverse_kl_topk`。
  仅在 `kl_family=verl` 时按此值生效；`kl_family=sdpo` 时会被覆盖为 `sdpo_alpha_kl_topk` 或 `sdpo_alpha_kl_sampled`。
- **`topk`**（int，`> 0`，默认 `32`）：教师 top-K 截断的 K 值，仅 top-K 类 loss（`forward_kl_topk` / `reverse_kl_topk` / `sdpo_alpha_kl_topk`）使用。
- **`use_task_rewards`**（bool，默认 `true`）：`true` 时蒸馏 loss 与任务 reward 并行；`false` 时纯蒸馏，忽略 RL reward。
- **`distillation_loss_coef`**（float，`>= 0`，默认 `1.0`）：蒸馏 loss 与任务 loss 相加时的权重。
- **`loss_max_clamp`**（float 或 `null`，默认 `10.0`）：蒸馏 loss 逐 token 绝对值上界，防止极端 token 主导梯度；设 `null` 关闭。
- **`log_prob_min_clamp`**（float 或 `null`，默认 `-10.0`）：把 logprob 下截到该值，避免 `log 0` 造成 inf/nan；设 `null` 关闭。
- **`use_policy_gradient`**（bool，默认 `true`）：是否把蒸馏目标当作 PG surrogate（套 PPO ratio + clip）。`false` 走直接监督式优化；`loss_mode=k1` 必须为 `true`。
- **`policy_loss_mode`**（str，仅 `vanilla`，默认 `vanilla`）：PG 形式下的 surrogate 类型，当前仅实现标准 PPO surrogate，其它值抛 `NotImplementedError`。仅 `use_policy_gradient=true` 时读取。
- **`clip_ratio` / `clip_ratio_low` / `clip_ratio_high`**（float，`> 0`，默认均 `0.2`）：PPO ratio 的 clip 半径。仅 `use_policy_gradient=true` 时读取。

SDPO 子组 `distillation.distillation_loss.sdpo.*`（仅 `kl_family=sdpo` 时读取）：

- **`alpha`**（float，`[0.0, 1.0]`，默认 `1.0`）：alpha-KL 的插值系数，控制 forward↔reverse KL 取向。常用 `0.0`（forward KL）、`0.5`（JSD）、`1.0`（reverse KL）。
- **`mode`**（str，`topk` / `full` / `sampled`，默认 `topk`）：alpha-KL 的估计方式。`topk` 在教师 top-K 上做精确 alpha-KL；`sampled` 用学生采样到的单 token 估计（退化为 k3 反向 KL），仅 `alpha=1.0` 合法；`full` 全词表，尚未实现。
- **`tail`**（str，`add` / `renorm` / `drop`，默认 `add`）：教师 top-K 之外剩余概率的处理。`add` 把 `1−Σp_i` 合并为虚拟 K+1 桶；`renorm` 把 top-K 重新归一化到 1；`drop` 直接丢掉 tail。仅 `mode=topk` 生效。
- **`ratio_clip`**（float 或 `null`，`> 0`，默认 `null`）：每 token 重要性权重 `exp(s − s_old)` 的上截阈值，抑制单 token 主导。仅 `mode=topk` 生效。

`reverse_kl_topk` 专用字段（只在 `loss_mode=reverse_kl_topk` 下读取）：

- **`norm_to_one_for_kl`**（bool，默认 `true`）：是否把教师 top-K 概率归一化到 1 再算 KL，等价于 SDPO `tail=renorm`。
- **`clip_log_ratio`**（bool，默认 `false`）：把 `log p_student − log p_teacher` clamp 到 `[-5, 5]`，防数值爆炸。
- **`use_tail_sampling`**（bool，默认 `false`）：用学生采样到的 token 作为 K+1 位补回，对其在 top-K 外的位置加 L2，补偿 top-K 截断偏差。
- **`use_kl_iw`**（bool，默认 `false`）：给 tail-sampling 的 L2 项乘以重要性权重 `exp(log π − log π_old)`。
- **`kl_iw_clip_lower` / `kl_iw_clip_upper`**（float 或 `null`，默认 `null`）：上述重要性权重的下/上截阈值。
- **`opd_mask_special_tokens`**（bool，默认 `false`）：是否在 loss 中 mask 掉每条 response 首个特殊 token，规避教师-学生 special token 不一致。OPSD 词表共享时启用会被忽略并 warning。
- **`opd_mask_first_tokens`**（list[str]，默认 `['<', 'think', '<|im_end|>']`）：上一项启用时要 mask 的 token 字面量列表，启动时由 tokenizer 转成 token ids。

### 3.3 OPSD 专用 `distillation.self_distill.*`

仅 `distillation.mode=self` 时读取；在 `distillation.mode=external` 下显式设置这一组任何字段都会被拒绝（trainer 启动时报错），因为外挂教师是独立进程，trainer 无法对它做 EMA / 周期硬拷贝 / trust-region 更新。

- **`dataloader`**（str，默认 `null`，**必填**）：teacher dataloader 的 importable FQN，形如 `pkg.module:Class`。类必须继承 `OfflineTeacherDataloader` 或 `OnlineTeacherDataloader`，用来定义每个样本如何被改写成教师 prompt。
- **`dataloader_kwargs`**（dict，默认 `{}`）：原样透传到 dataloader 的 `__init__(**kwargs)`，方便传入自定义超参。
- **`teacher_update`**（str，`ref` / `ema` / `progressive` / `trust_region`，默认 `ema`）：教师权重更新策略。
  - `ref`：教师永远固定，等同标准 KD；
  - `ema`：`ref ← decay · ref + (1 − decay) · actor`，平滑跟随学生；
  - `progressive`：每隔若干步硬拷贝学生权重；
  - `trust_region`：当 KL(student‖teacher) 达阈值后才硬拷贝。
- **`ema_decay`**（float，`(0.0, 1.0)`，默认 `0.999`）：EMA 衰减系数，值越大教师越滞后。仅 `teacher_update=ema` 生效。
- **`teacher_update_rate`**（float，`>= 0`，默认 `0.0`）：EMA 的等价参数。非 0 时覆盖 `ema_decay`，对应 `ema_decay = 1 − teacher_update_rate`。
- **`teacher_update_interval`**（int，`> 0`，默认 `0`）：progressive 模式下硬拷贝学生权重的间隔步数，progressive 模式必填。
- **`trust_region_threshold`**（float，`>= 0`，默认 `0.0`）：trust_region 模式下触发硬拷贝的 KL 阈值。
- **`truncation`**（str，`left` / `right` / `error`，默认 `right`）：教师 prompt 超过 `data.max_prompt_length` 时的截断策略，分别表示左截/右截/直接报错。

---

## 4. 当前限制

- top-K kernel（forward/reverse/sdpo_alpha_kl_topk）仅 FSDP 后端有 OPSD 端的 teacher top-K 抽取；Megatron 后端需走 estimator 路径。
- `sdpo.mode=full`（全词表 alpha-KL）尚未接通。

---

## 5. 常见问题

| 现象 | 解决 |
|---|---|
| `self_distill.dataloader` 为空 | 必须设置，格式 `pkg.module:Class` |
| `... is not a subclass of OfflineTeacherDataloader or OnlineTeacherDataloader` | 你的类必须继承其中一个基类 |
| `ValueError: teacher_update must be one of ...` | 检查拼写（`ref`/`ema`/`progressive`/`trust_region`） |
| `OPSD progressive teacher requires teacher_update_interval > 0` | progressive 模式必须设 `teacher_update_interval` |
| `OPSD ema teacher requires 0 < ema_decay < 1` | 调 `ema_decay` 到 (0, 1) |
| `NotImplementedError: ... topk distillation losses` (Megatron) | 改用 estimator 路径（`loss_mode=k3` 等） |
| 蒸馏 loss 一直为 0 | 多半是 dataloader 全部返回 `skip=True`，检查 `extra_info` 字段 |
| Teacher prompt 被截断 | 提高 `data.max_prompt_length` 或改 `truncation=left` |
| `KeyError: sampled_teacher_logprob` | 启用了 `use_tail_sampling=True` 但走的不是 OPSD topk 路径；OPD 外挂教师需自行 plumbing 该字段 |
