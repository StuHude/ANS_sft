# ANS 项目交接说明

## 项目概览
本项目可视为两部分：
1) **原始 Sa2VA SFT 代码（稳定正确）**：
- 训练配置在 `/data/xyc/ANS/projects/llava_sam2/configs/`，如 `sa2va_4b.py`。
- 数据集代码在 `/data/xyc/ANS/projects/llava_sam2/datasets/`（不含 `external_loaders`）。
2) **新增 mask_caption_sft 研究代码（当前开发部分）**：
- 训练主逻辑：`/data/xyc/ANS/projects/llava_sam2/mask_caption_sft/train_pseudo_gumbel_v2.py`
- 常用启动脚本参考：`/data/xyc/ANS/run_pseudo_gumbel_v2_8gpu_dam_dual_loop_save250.sh`
  - 实际训练建议加 `--disable_internal_eval`，避免保存后自动 eval 中断训练。

## 训练逻辑（核心）
### 循环1（原始 4 数据集）
数据集：SAV / SA1B / OpenImage / RefCOCO。
- EMA teacher：`pseudo_ids = ema.generate(image1, mask1)`（stop-grad，T=固定长度）
- Mask 伪 token：随机 mask 比例 0.25
- Trainable 1：`image1+mask1+pseudo_masked` 得到 logits
- ST-Gumbel：top-k（常用 512）+ `gumbel_softmax(hard=True)` → `text_embeds`
- Trainable 2：`image2 + inputs_embeds=text_embeds` 预测 mask2
- Loss：mask loss（BCE + dice），梯度必须穿过 gumbel 与第一次 forward
- RefCOCO 仅用于指代分割；SA1B/OpenImage 没有 image2 时令 image2= image1、mask2=mask1。

### 循环2（DAM 双循环）
数据集：Describe Anything Dataset（SAV, COCOStuff, LVIS, Mapillary, OpenImages, PACO）。
- A：`image + caption` → 预测 mask'（分割任务，使用与循环1一致的分割模板）
- B：`image + mask'` → 预测 caption'（描述任务，使用与循环1一致的描述模板）
- Loss2：CE(caption', caption)
- 训练时循环1与循环2交替，做梯度手术：将循环2梯度投影到循环1梯度正交后再更新；总 loss = loss1 + β * loss2（β=0.5）。

**重要修正**：循环2中不再因为 mask' 为空而跳过 Step B。即使 mask 为空也会计算 CE，避免 `dam_caption_ce` 频繁为 0。

## 训练要点
- 仅训练：proj(mlp1/text_hidden_fcs)、LLM LoRA、SAM2 decoder，其余冻结。
- 必须使用 `inputs_embeds` 走第二次 forward，禁止离散 token chain。
- 建议训练时加 `--disable_internal_eval`（避免保存后自动 eval 打断训练）。
- 不要修改模型结构目录：`projects/llava_sam2/models/` 与 `projects/llava_sam2/hf/models/`。

## Eval 范式（重要）
当前经验：**直接保存的 pth 需转 HF 再 eval**，且必须避免覆盖 `embed/lm_head`，否则 `[SEG]` 生成会崩。

推荐流程：
1) 保存 pth（默认已避免导出 embed/lm_head）。
2) 使用 `tools/convert_ckpt_to_hf_simple.py` 将 pth 转 HF。
3) 用 `projects/llava_sam2/evaluation/refcoco_eval.py` 做 8 卡评测。

对比的评测模式（需分别生成/过滤权重）：
- full（不含 embed/lm_head）
- LoRA-only
- SAM2 + text_hidden_fcs-only
- LoRA + SAM2 + text_hidden_fcs
- proj(mlp1)-only

常用 eval 参数示例：`--max_samples 100 --max_texts_per_image 1 --max_new_tokens 512`。

## 当前进展与注意事项
- 双循环训练已实现并可启动；训练时建议关闭内部 eval。
- DAM splits 已启用：SAV、COCOStuff、LVIS、Mapillary、OpenImages、PACO。
- 任何新服务器运行前，先确认数据路径与 `--model_path / --pretrained_pth`。
