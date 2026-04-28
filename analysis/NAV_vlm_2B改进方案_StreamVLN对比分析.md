# StreamVLN 对比下的 NAV_vlm 2B 问题分析与改进计划

日期：2026-04-28  
目标：基于 `external/NAV_vlm` 继续改进 Qwen3-VL-2B 的 R2R 离散 VLN 训练，不直接融合 StreamVLN 主体代码。

---

## 1. 总体判断

当前不建议把 StreamVLN 的模型代码直接融合进 NAV_vlm。

原因：

1. StreamVLN 是连续环境的低层动作预测框架，核心动作是 `TURN LEFT / TURN RIGHT / MOVE FORWARD / STOP`。
2. NAV_vlm 是离散拓扑图上的候选点选择框架，核心动作是 `(node_X)`、`(history_node_X)`、`(node_0)`。
3. 两者的动作空间、数据结构、视觉输入组织、训练数据分布都不同，直接融合成本高，收益不确定。

更合理的方向是：

1. 保留 NAV_vlm 的核心思路：Qwen3-VL + CoT + 拓扑图 + 候选点选择。
2. 借鉴 StreamVLN 的训练范式：短输出、强闭环数据、历史记忆、通用视觉数据正则。
3. 借鉴 ETP-R1 的候选点视觉结构：候选方向、距离、候选局部视觉、visited/frontier 拓扑表达。
4. 当前阶段继续使用 Qwen3-VL-2B，先把数据、prompt、评估闭环和训练目标做干净；等 2B 方案稳定后再上 8B。

---

## 2. StreamVLN 代码分析

### 2.1 训练入口

主要入口：

- `streamvln/streamvln_train.py`
- `streamvln/dataset/vln_action_dataset.py`
- `streamvln/model/stream_video_vln.py`
- `llava/train/llava_trainer.py`

训练流程：

1. `streamvln_train.py` 解析 model/data/training 参数。
2. `make_supervised_data_module()` 创建 `VLNActionDataset`。
3. `VLNActionDataset` 从 `video_folder/*/annotations.json` 加载轨迹数据和 RGB 帧。
4. 每个样本包含：
   - 多帧图像；
   - 历史帧；
   - VLN instruction；
   - 未来若干步动作序列；
   - Qwen/LLaVA 风格对话文本。
5. `StreamVLNForCausalLM` 将 `<image>` 和 `<memory>` token 替换为视觉 embedding。
6. `LLaVATrainer` 完成训练。

注意：StreamVLN 训练数据不是直接读 `data/r2r/train.json.gz`。`train.json.gz` 是 Habitat/VLN-CE episode 定义，StreamVLN 训练实际使用的是已经转换好的 trajectory/video 数据。

### 2.2 StreamVLN prompt

StreamVLN 的导航 prompt 很短，核心在 `streamvln/dataset/vln_action_dataset.py`：

```text
You are an autonomous navigation assistant.
Your task is to <instruction>.
Devise an action sequence to follow the instruction using the four actions:
TURN LEFT (←) or TURN RIGHT (→) by 15 degrees,
MOVE FORWARD (↑) by 25 centimeters, or STOP.
```

之后每一步只是追加类似：

```text
you can see <image>.
```

如果有历史，则加入：

```text
These are your historical observations: <memory>.
```

输出是短动作序列，例如：

```text
↑↑←↑STOP
```

特点：

1. 文本规则短。
2. 动作 token 稳定。
3. 不要求生成长 CoT。
4. 视觉 token 占主要上下文。
5. 低层动作空间固定，不需要复制动态 label。

### 2.3 StreamVLN 视觉处理

核心文件：`streamvln/model/stream_video_vln.py`。

关键函数：

- `encode_rgbd()`
- `prepare_inputs_labels_for_multimodal()`
- `generate()`

实际流程：

1. 多帧 RGB 输入 vision tower。
2. vision tower 输出 patch features。
3. `mm_projector` 映射到 LLM hidden size。
4. 当前帧通过 `<image>` token 注入。
5. 历史帧通过 `<memory>` token 注入。
6. 历史帧会经过 2D pooling，减少视觉 token 数。
7. 生成时维护 cache，把历史输入和当前输入拼接起来。

重要细节：

虽然 StreamVLN eval/DAgger 代码里会传 `depths / poses / intrinsics`，但当前 `encode_rgbd()` 主要使用 RGB image features；depth、pose、intrinsic 在这个模型实现里没有真正形成显式 3D 空间建模。也就是说，StreamVLN 的强点主要来自：

1. 多帧视频流；
2. `<memory>` 历史视觉压缩；
3. 短动作输出；
4. 大规模训练数据；
5. DAgger 闭环纠错数据；
6. 通用视觉语言数据混训。

### 2.4 StreamVLN 数据规模和训练范式

论文 `analysis/2507StreamVLN.pdf` 中的重要信息：

1. 基础模型：LLaVA-Video 7B / Qwen2-7B。
2. 第一阶段：oracle VLN 轨迹训练。
3. 第二阶段：DAgger + 通用视觉语言数据混训。
4. 数据包括：
   - R2R / R2R-EnvDrop / RxR oracle 轨迹；
   - ScaleVLN / HM3D 子集；
   - DAgger corrective samples；
   - VideoQA / ScanQA；
   - MMC4。
5. 训练成本约 1500 A100 GPU hours。

论文消融显示：

1. 只用第一阶段 oracle VLN，性能明显低于完整版本。
2. 加 DAgger 后提升明显。
3. 加 RxR、ScaleVLN、VideoQA/MMC4 后继续提升。
4. memory context 太少或太多都会下降，适中窗口最好。

这说明 StreamVLN 不是靠一个简单 prompt 或模型结构赢的，而是完整训练 recipe 共同作用。

---

## 3. NAV_vlm 2B 当前代码理解

### 3.1 核心结构

主要目录：`external/NAV_vlm`。

关键模块：

- `sft_full/common/prompts.py`：SFT prompt。
- `sft_full/common/dataset.py`：CoT/Gemini 数据集与 collator。
- `sft_full/models/policy.py`：Qwen3-VL 加载和 forward loss。
- `sft_full/train_sft_full.py`：SFT 训练入口。
- `sft_full/eval_discrete_full.py`：离散闭环评估。
- `data_engine/generate_cot_v2.py`：CoT 数据生成。
- `data_engine/utils_v2.py`：候选点、图结构、图像标注。
- `grpo_full/*`：后续在线/强化学习相关代码。

### 3.2 当前训练范式

当前主线是：

1. 使用 R2R 路径和 connectivity graph 生成每一步候选点。
2. 当前图像是三视图 pano，并且候选点用数字标出来。
3. prompt 中提供 instruction、视觉观察、拓扑图、历史、候选、回退候选、stop。
4. 使用外部强模型生成 CoT。
5. SFT 训练 Qwen3-VL-2B 输出：

```text
<think>...</think>

<answer>(node_X)</answer>
```

或：

```text
<answer>(node_X)</answer>
```

### 3.3 候选点视觉标注

需要修正此前误解：NAV_vlm 当前输入给 VLM 的图像不是普通单帧图，而是三视图 pano，并带候选点标记。

用户给出的例子类似：

- Left (-120)
- Front (0)
- Right (+120)
- 候选点以数字圆圈标在图中
- 下方有角度刻度

本地代码中相关位置：

1. `data_engine/utils_v2.py` 的 `WaypointAnnotator`：用于数据生成阶段，在图中画候选编号。
2. `sft_full/common/visual_prompt.py` 的 `annotate_candidate_image()`：用于评估/调试阶段，根据候选描述给当前图补标记。
3. `sft_full/eval_discrete_full.py`：评估时会对当前图调用 `annotate_candidate_image()`。

需要注意的代码级风险：

1. `generate_cot_v2.py` 里 CoT 生成阶段调用：

```python
annotator.annotate_image(image, main_candidates, None if is_terminal_step else next_gt_vp)
```

这会在 CoT 生成时把 GT 候选高亮成特殊样式。如果强模型生成 CoT 时看到 GT 高亮，CoT 数据会有潜在答案泄漏。

2. `result` 中保存的是 `current_image_path`，不是 `annotated_img` 的路径。也就是说，从本地代码看，SFT dataloader 是否实际加载了带标注图，需要用 debug batch 确认。用户确认“当前环境每个阶段送入 VLM 都有标注”，因此后续不把“没有候选标注”作为问题，但仍建议做一次训练输入审计，确认训练、评估、CoT 生成三阶段的标注样式完全一致。

---

## 4. NAV_vlm 2B 当前主要问题

### 4.1 数据不是简单少，而是闭环分布不足

80K 单步 CoT 数据不算很少，但它主要来自 oracle 路径状态。

问题在于：

1. 训练样本多是“正确路径上的状态”。
2. 闭环评估时模型只要走错一步，就进入训练集中很少出现的 off-route 状态。
3. 后续图结构、history、candidate 分布都会变。
4. 模型没有学过如何恢复。

这就是为什么单步 SFT 看起来合理，但 closed-loop SR 可能只有 36%。

StreamVLN 的 DAgger 数据正是解决这个问题：让模型在自己犯错后的状态里学习如何回到正确路径。

### 4.2 2B 承担的任务太重

当前 NAV_vlm 让 Qwen3-VL-2B 同时完成：

1. 读三视图候选标注图。
2. 理解自然语言 instruction。
3. 理解拓扑图。
4. 记住历史节点。
5. 判断是否 stop。
6. 判断是否 backtrack。
7. 生成长 `<think>`。
8. 最后复制动态 label。

2B 模型容量有限，长 CoT 会让训练目标变得更难：

1. 大量 loss 花在学习解释文本，而不是动作。
2. 生成越长，格式错和解析错概率越高。
3. 长 CoT 容易在闭环中引入幻觉。
4. 动态 label `(node_1)` 的语义每一步都变，模型必须依赖 prompt 复制，而不是学稳定动作 token。

### 4.3 Prompt 文本过重

当前 `sft_full/common/prompts.py` 中 prompt 包含：

1. system prompt 的多条规则。
2. Visual Stream Semantics。
3. Visual Observations。
4. Graph。
5. Recent Moves。
6. Current Candidates。
7. Backtrack Candidates。
8. Stop。
9. 重复 instruction。

相比 StreamVLN 的短 prompt，NAV_vlm prompt 更像一个长文本推理任务。

对 2B 的影响：

1. 上下文被文本占满，视觉 token 的相对作用下降。
2. 模型容易学到“根据候选文字和历史模板猜答案”，而不是基于图像决策。
3. prompt 长度变动大，训练不稳定。
4. 每一步生成长 CoT，计算成本和错误率都高。

### 4.4 视觉已经有候选标注，但还可以更候选中心化

当前视觉输入已经把候选点编号画在三视图图像上，这一点是正确的。

但仍有提升空间：

1. 图中候选点只是一个编号点，模型需要自己在整张 pano 中找到对应区域。
2. 候选点附近的局部视觉可能太小，2B 不一定能稳定注意。
3. 如果候选点很多，编号和候选文本的对应关系可能仍然有负担。
4. 历史图像和当前图像混在一起时，当前候选视觉可能被稀释。

ETP-R1 的可借鉴点不是模型结构，而是候选中心表达：

1. 12-view panorama。
2. 每个候选有对应的 `cand_rgb`。
3. 每个候选有 `cand_depth`。
4. 每个候选有 angle/distance feature。
5. 模型显式按候选比较，而不是只看整张图。

对于 VLM 版本，可以改成：

1. 保留三视图标注图。
2. 额外给每个候选一个 crop/panel。
3. prompt 中严格按候选顺序引用 crop。
4. 对复杂路口优先使用候选 crop。

### 4.5 可能存在 GT 高亮泄漏

这是必须优先审计的问题。

`WaypointAnnotator.annotate_image()` 支持 `gt_next_id`，如果传入 GT，会对正确候选使用特殊绿色粗圈或星标。

如果这个图像被用于 CoT 生成，强模型生成的 CoT 可能只是看到了 GT 高亮，而不是根据 instruction 和环境推理。

后果：

1. CoT 文本看起来合理，但其实被图像泄漏污染。
2. SFT 模型学到的 reasoning 质量不可靠。
3. 如果 SFT/eval 阶段没有同样 GT 高亮，训练和评估分布不一致。
4. 这会严重解释“训练数据不少，但效果不好”的情况。

处理原则：

1. 训练、CoT 生成、评估中都只能有普通候选编号。
2. GT 只能存在于 JSON 标签中，不能以视觉样式泄漏给模型。
3. debug 可视化可以高亮 GT，但不能进入 VLM 输入。

### 4.6 8B 训练当前不是效果差，而是没有成功训练

`external/NAV_vlm/wrong.md` 显示 Qwen3-VL-8B full finetune 在 optimizer step 初始化 Adam state 时 OOM。

这说明：

1. 不能用这次记录判断 8B 性能。
2. 当前阶段坚持 2B 是合理的。
3. 后续上 8B 应使用 LoRA/QLoRA、ZeRO-3 offload、短 prompt、低图像数，而不是 full finetune 起步。

### 4.7 评估中还有一些会拉低 SR 的工程风险

当前不是优先项，但后续需要处理：

1. `run_discrete_eval_full.py` 默认 `do_sample=1, temperature=0.5, top_p=0.9`，闭环导航评估应优先使用 greedy。
2. `eval_discrete_full.py` 中 candidates 为空时会强制 stop，可能造成未到 max step 提前结束。
3. parser fallback 可能把解释里的最后一个 node 误当答案。
4. stop precision/recall 需要单独统计。
5. reached-goal 但未 stop 的 episode 和 premature stop episode 应拆开看。

---

## 5. NAV_vlm 2B 改进目标

短期目标不是直接达到 70% SR，而是把问题拆清楚：

1. 单步动作准确率是否足够高。
2. 闭环掉分主要来自 off-route、stop、parser、候选缺失还是视觉误判。
3. CoT 是否真的帮助 2B，还是拖累动作学习。
4. 候选 crop 是否能提升复杂路口选择。
5. DAgger-light 是否能显著提升 closed-loop SR。

建议阶段目标：

1. Stage A：保证数据和评估干净。
2. Stage B：建立 answer-only 和 short-CoT baseline。
3. Stage C：加入候选中心视觉。
4. Stage D：加入 DAgger-light recovery 数据。
5. Stage E：稳定后再迁移到 8B LoRA/QLoRA。

---

## 6. 详细改进计划

### 6.1 Stage A：数据与输入审计

优先级最高。

需要做：

1. 审计 CoT 生成输入图：
   - 随机抽样保存 VLM 请求图像。
   - 确认候选点只有普通编号。
   - 确认没有 GT 星标、粗绿色圈、特殊颜色。

2. 审计 SFT collator 输入：
   - 使用 `debug_mode` dump batch。
   - 保存模型实际收到的全部 image slots。
   - 保存对应 prompt。
   - 保存 supervised label span。

3. 审计 eval 输入：
   - 保存评估时每一步 VLM 实际输入图。
   - 确认标注样式和 SFT 一致。

4. 统计数据质量：
   - `answer_ok` 比例；
   - `cot_answer_mismatch` 比例；
   - stop 样本占比；
   - backtrack 样本占比；
   - candidate count 分布；
   - 每条样本图像数量分布；
   - prompt token 长度分布；
   - supervised token 长度分布。

预期结论：

如果发现 GT 视觉泄漏，必须先修数据生成，再重新生成 CoT 或至少重新生成受污染部分。

### 6.2 Stage B：压缩 prompt

新增一个 compact prompt 模式，不直接删除旧 prompt。

建议 prompt：

```text
Instruction:
{instruction}

Images:
{visual_observations}

Recent Moves:
{recent_moves}

Candidates:
{candidate_descriptions}

Backtrack:
{backtrack_descriptions}

Stop:
- (node_0)

Output:
<think>brief evidence-based reason</think>
<answer>one legal label</answer>
```

压缩原则：

1. 删除重复 instruction。
2. 删除长 Visual Stream Semantics。
3. Graph 只保留当前决策有关内容：
   - current node；
   - visited 简表；
   - unexplored frontier 简表；
   - backtrack 候选。
4. Recent Moves 最多 3 条。
5. 候选描述保留 label、方向、角度、距离、visited/frontier 状态。
6. system prompt 从长规则改成短规则。

短 system prompt：

```text
You are a VLN navigation policy.
Choose exactly one legal action label.
Use the current annotated image first, then instruction and topology.
Stop only when the target is reached.
```

### 6.3 Stage C：改训练目标，降低 2B 负担

需要建立三个 baseline：

#### Baseline 1：Answer-only

输出：

```text
<answer>(node_X)</answer>
```

目的：

1. 判断 2B 是否能学会动作选择本身。
2. 排除长 CoT 对动作学习的干扰。
3. 给 closed-loop 提供更稳定解析。

#### Baseline 2：Short-CoT

输出：

```text
<think>短理由，2-4 句。</think>
<answer>(node_X)</answer>
```

要求：

1. think 限制长度。
2. 不允许长篇环境复述。
3. 不允许重复候选列表。
4. answer token loss 权重大于 think token。

#### Baseline 3：Adaptive-CoT

简单步骤 answer-only，复杂步骤 short-CoT。

复杂步骤定义：

1. candidate count >= 3；
2. 需要 stop；
3. 需要 backtrack；
4. 当前候选方向分布相近；
5. 历史中存在回环风险。

推荐先做 Baseline 1 和 Baseline 2，不要一开始做复杂 adaptive。

### 6.4 Stage D：候选中心视觉增强

当前已有三视图候选标注图，后续增强不是“从无到有”，而是让 2B 更容易关注候选区域。

建议新增：

1. `candidate_crops`：
   - 按候选点坐标裁剪局部图；
   - 每个候选一个 crop；
   - crop 顺序与 `(node_1), (node_2), ...` 完全一致。

2. prompt 中加入：

```text
Current panorama: <image>
Candidate views:
- (node_1): <image>, go left, 1.2m
- (node_2): <image>, go forward, 0.8m
```

3. 控制图像数量：
   - 2B 阶段最多保留当前 pano + top-K candidate crops + 关键历史图。
   - K 建议 3 或 4。
   - 如果候选太多，优先保留 oracle label、近距离候选、方向差异大的候选；评估时不能用 oracle 筛选，只能用固定规则。

4. 与 ETP-R1 的对应关系：
   - ETP-R1 的 `cand_rgb` 对应 VLM candidate crop。
   - ETP-R1 的 `cand_angle_fts` 对应 prompt 中的 angle/direction。
   - ETP-R1 的 `cand_distances` 对应 prompt 中的 distance。
   - ETP-R1 的 pano feature 对应当前三视图 pano。

注意：

不要引入 ETP-R1 的非 VLM policy。只借鉴候选视觉组织方式。

### 6.5 Stage E：DAgger-light recovery 数据

这是提升 closed-loop SR 的关键。

流程：

1. 用当前 2B 模型在 train split 或 seen scenes 上闭环 rollout。
2. 每一步记录：
   - 当前 viewpoint；
   - history；
   - candidate list；
   - model prediction；
   - 是否偏离 oracle path；
   - 到 goal 的 shortest next viewpoint。
3. 用 shortest path teacher 给当前 off-route 状态重新标注 `gt_label`。
4. 生成 recovery SFT 数据。
5. recovery 数据优先 answer-only 或 short-CoT，不先追求长 CoT。
6. 按比例混入原始 oracle 数据。

推荐混合比例：

1. 初始：oracle 70%，recovery 30%。
2. 如果模型变得激进或 stop 变差，增加 oracle/stop 样本。
3. 如果闭环仍容易走不回来，提高 recovery。

### 6.6 Stage F：通用视觉语言正则

StreamVLN 混入 VideoQA、ScanQA、MMC4 的目的之一是防止导航 SFT 让 VLM 丢掉通用视觉能力。

NAV_vlm 2B 可以轻量加入：

1. 少量室内 VQA；
2. ScanQA/SQA3D 类问答；
3. 只占 5%-10%；
4. 不需要和导航动作混在同一个格式；
5. 主要用于保视觉理解和语言能力。

这不是第一优先级，但 full finetune 或长 epoch 训练时很重要。

---

## 7. 推荐实验顺序

### Experiment 0：输入一致性审计

目的：确认训练、CoT 生成、评估的图像标注一致，且无 GT 泄漏。

输出：

1. 20 个训练 batch 可视化。
2. 20 个 CoT 生成请求样本。
3. 20 个 eval 请求样本。
4. 每个样本保存 prompt、image slots、gt_label、candidate list。

### Experiment 1：Answer-only baseline

目的：判断动作选择上限。

设置：

1. Qwen3-VL-2B。
2. compact prompt。
3. 只训练 `<answer>`。
4. greedy eval。

观察：

1. val-unseen 单步 accuracy。
2. closed-loop SR/SPL。
3. parse failure 是否接近 0。
4. stop precision/recall。

### Experiment 2：Short-CoT baseline

目的：判断短 CoT 是否比 answer-only 更好。

设置：

1. compact prompt。
2. `<think>` 限长。
3. answer loss 加权。
4. greedy eval。

观察：

1. 单步 accuracy 是否提升。
2. closed-loop 是否提升。
3. 是否出现更多格式错误。
4. 是否有 premature stop。

### Experiment 3：Candidate crop

目的：验证候选中心视觉是否改善复杂路口。

设置：

1. 当前三视图 pano 保留。
2. 加 top-K candidate crops。
3. prompt 与 image slots 顺序严格一致。

观察：

1. candidate count >= 3 的单步 accuracy。
2. 复杂路口 closed-loop 成功率。
3. 图像数量增加后的显存和速度。

### Experiment 4：DAgger-light

目的：解决 closed-loop distribution shift。

设置：

1. 用 Experiment 1 或 2 的最佳 checkpoint rollout。
2. 收集 off-route 状态。
3. shortest path teacher 重新标注。
4. 混入 30% recovery 数据再 SFT。

观察：

1. closed-loop SR/SPL。
2. off-route recovery rate。
3. loop rate。
4. stop precision/recall。

---

## 8. 评估指标需要拆细

不要只看 SR。

每次 eval 至少输出：

1. `closed_loop_success`
2. `SPL`
3. `success_ignore_stop`
4. `reached_goal_but_no_stop`
5. `premature_stop`
6. `parse_fail_rate`
7. `invalid_label_rate`
8. `candidate_hit_rate`
9. `no_candidates_force_stop_count`
10. `stop_precision`
11. `stop_recall`
12. `avg_final_distance`
13. `avg_steps`
14. `loop_count`
15. `backtrack_action_rate`

这样才能知道 36% 到底卡在哪里。

---

## 9. 需要修改的主要位置

### 9.1 Prompt

文件：

- `external/NAV_vlm/sft_full/common/prompts.py`
- `external/NAV_vlm/data_engine/cot_prompts.py`

任务：

1. 增加 compact prompt 模式。
2. 保留旧模式，便于对照实验。
3. 删除重复 instruction。
4. 限制 recent moves 和 graph 文本长度。

### 9.2 Dataset / Collator

文件：

- `external/NAV_vlm/sft_full/common/dataset.py`

任务：

1. 支持 answer-only。
2. 支持 short-CoT。
3. 支持 answer token 加权。
4. dump image slots 和 prompt 便于审计。
5. 支持 candidate crops 输入。

### 9.3 CoT 生成

文件：

- `external/NAV_vlm/data_engine/generate_cot_v2.py`
- `external/NAV_vlm/data_engine/utils_v2.py`

任务：

1. 确保 VLM 输入图不含 GT 高亮。
2. 生成 short-CoT 数据。
3. 保存候选 crop 元数据。
4. 统计 answer_ok 和样本质量。

### 9.4 Eval

文件：

- `external/NAV_vlm/sft_full/eval_discrete_full.py`
- `external/NAV_vlm/sft_full/run_discrete_eval_full.py`

任务：

1. 默认 greedy eval。
2. 输出 failure breakdown。
3. 保存失败样本 prompt/images。
4. 支持 compact prompt 和 candidate crops。
5. 统计 stop precision/recall。

### 9.5 DAgger-light

文件：

- 可以新增到 `external/NAV_vlm/data_engine/` 或复用 `grpo_full/envs.py` 的环境逻辑。

任务：

1. 闭环 rollout。
2. shortest path teacher 标注。
3. 生成 recovery JSONL。
4. 与 oracle SFT 数据混训。

---

## 10. 结论

NAV_vlm 的核心方向是合理的：用 Qwen3-VL 学习 CoT，再结合拓扑图做离散 VLN 候选选择。

当前 36% SR 的主要原因更可能是：

1. 缺 closed-loop recovery 数据；
2. 2B 模型承受了过长 prompt 和过长 CoT；
3. 训练目标没有突出最终 action；
4. 可能存在 CoT 生成阶段 GT 高亮泄漏；
5. 评估指标没有拆开，导致不知道具体失败类型；
6. 候选视觉虽已标注，但还不是 ETP-R1 那种候选中心表达。

下一步最推荐的路线：

1. 先做输入一致性和 GT 泄漏审计。
2. 做 compact prompt。
3. 建立 answer-only baseline。
4. 建立 short-CoT + answer loss 加权 baseline。
5. 增加 candidate crops。
6. 做 DAgger-light recovery 数据。
7. 2B 稳定后，再考虑 8B LoRA/QLoRA。

这条路线比直接改 StreamVLN 更适合当前项目，也更容易定位每一个改动到底带来了什么收益。
