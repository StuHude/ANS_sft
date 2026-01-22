"""
Core training loop implementations for Pseudo Token + Gumbel-Softmax Training.

This file contains the critical training logic separated for clarity.
"""

import random
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import GenerationConfig
import os


def _count_vp_tokens_from_prompt_mask(prompt_mask: torch.Tensor, num_img_tokens: int, device: torch.device) -> int:
    """
    Count K for "<vp>{<IMG_CONTEXT>*K}</vp>" in a way that is consistent with the vp_embeds
    construction below (which may resize the mask to match the ViT token grid).
    """
    side = int(num_img_tokens ** 0.5)
    if side * side != int(num_img_tokens):
        raise ValueError(f"num_img_tokens must be a perfect square, got {num_img_tokens}")

    m = prompt_mask
    if m.ndim == 3:
        # (1,G,G) -> (G,G)
        m = m[0]
    if m.ndim != 2:
        raise ValueError(f"prompt_mask must be 2D or 3D, got shape={tuple(prompt_mask.shape)}")

    m = m.float().to(device=device).unsqueeze(0).unsqueeze(0)  # (1,1,G,G)
    if m.shape[-2:] != (side, side):
        m = F.interpolate(m, size=(side, side), mode='nearest')
    return int(m.bool().sum().item())


def _global_max_len(local_len: int, device: torch.device) -> int:
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([int(local_len)], device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())
    return int(local_len)


def _pad_2d_right(x: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    if x.shape[1] >= target_len:
        return x
    pad = torch.full(
        (x.shape[0], target_len - x.shape[1]),
        pad_value,
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat([x, pad], dim=1)


def generate_pseudo_tokens_with_ema(ema_model, pixel_values, prompt_masks, tokenizer, max_caption_len, device):
    """
    Step 1: EMA model generates pseudo tokens from image + mask.

    Args:
        ema_model: EMA copy of the model (stop-grad)
        pixel_values: [B, 3, 448, 448] - ImageNet normalized for InternVL
        prompt_masks: [B, 16, 16] - Visual prompt masks [0, 1]
        tokenizer: tokenizer
        max_caption_len: maximum caption length
        device: device

    Returns:
        pseudo_toks: [B, max_caption_len] token IDs
    """
    batch_size = pixel_values.shape[0]

    # Images and masks are already in the correct format from dataset
    images_448 = pixel_values  # Already (B, 3, 448, 448) normalized
    # prompt_masks already (B, 16, 16)

    # Build input prompt matching DescribeAnythingReferringDataset (sa2va_4b.py) + phi3_chat template.
    # Input (after <image> replacement):
    #   "<img>{IMG_CONTEXT*256}</img>\n"
    #   "There are 1 part regions in the picture: region1<vp>{IMG_CONTEXT*K}</vp>.\n"
    #   "Please generate a detailed description for the given image region."
    # Wrapped by phi3_chat:
    #   "<|user|>\n{input}<|end|>\n<|assistant|>\n"
    IMG_CONTEXT = '<IMG_CONTEXT>'
    NUM_IMG_TOKENS = 256  # 16x16 grid

    input_ids_list = []
    for i in range(batch_size):
        K = _count_vp_tokens_from_prompt_mask(
            prompt_mask=prompt_masks[i],
            num_img_tokens=NUM_IMG_TOKENS,
            device=device,
        )
        img_str = f'<img>{IMG_CONTEXT * NUM_IMG_TOKENS}</img>'
        human_input = (
            f"{img_str}\n"
            f"There are 1 part regions in the picture: region1<vp>{IMG_CONTEXT * K}</vp>.\n"
            "Please generate a detailed description for the given image region."
        )
        prompt = f"<|user|>\n{human_input}<|end|>\n<|assistant|>\n"
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids_list.append(ids)

    # Pad to max length
    max_len = max(len(ids) for ids in input_ids_list)
    max_len = _global_max_len(max_len, device=device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    for i, ids in enumerate(input_ids_list):
        input_ids[i, :len(ids)] = torch.tensor(ids, device=device)
        attention_mask[i, :len(ids)] = True

    # Generate with EMA (greedy, no gradient).
    #
    # IMPORTANT:
    # - Do NOT call `ema_model.mllm.generate(...)` (InternVL wrapper).
    #   That wrapper forwards an argument named `return_dict` into HF `generate()`,
    #   which can end up in `model_inputs` and crash inside transformers with:
    #     "got multiple values for keyword argument 'return_dict'".
    #
    # We instead:
    # 1) Build base token embeddings from the LLM embedding layer.
    # 2) Compute vit/vp embeddings with the frozen vision encoder.
    # 3) Inject those embeddings into <IMG_CONTEXT> token positions.
    # 4) Call HF `language_model.generate()` directly with `inputs_embeds`.
    with torch.no_grad():
        img_context_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        if img_context_id is None or img_context_id < 0:
            raise RuntimeError("Tokenizer missing <IMG_CONTEXT> token id")

        gen_config = GenerationConfig(
            max_new_tokens=max_caption_len,
            do_sample=False,  # Greedy decoding
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        # Build pixel_values list in InternVL expected format (list of (T,3,H,W) or (3,H,W)).
        pixel_values_list = [images_448[i] for i in range(batch_size)]
        pv = [x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values_list]

        # Extract vit embeddings (vision encoder is frozen)
        vision_dtype = ema_model.mllm.model.vision_model.dtype
        concat_images = torch.cat([x.to(vision_dtype) for x in pv], dim=0).to(device)
        image_flags = (torch.sum(concat_images, dim=(1, 2, 3)) != 0).long()
        if (image_flags == 0).any():
            raise RuntimeError("Found all-zero image tile(s) after preprocessing (image_flags==0) in EMA generate")
        vit_embeds = ema_model.mllm.model.extract_feature(concat_images)
        vit_embeds = vit_embeds[image_flags == 1]
        if len(vit_embeds) != batch_size:
            raise RuntimeError(
                f"Unexpected vit batch size in EMA generate: {len(vit_embeds)} (expected {batch_size})"
            )

        # Build vp_embeds matching InternVL.generate semantics.
        prompt_masks_list = [prompt_masks[i:i + 1].to(device) for i in range(batch_size)]
        vp_overall_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        C = vit_embeds.shape[-1]
        vp_parts = []
        i_vp_img = 0
        for i_img in range(len(vit_embeds)):
            tile = vit_embeds[i_img].reshape(-1, C)
            vp_parts.append(tile)
            if bool(vp_overall_mask[i_img].item()):
                objects_prompt_masks = prompt_masks_list[i_vp_img].to(vit_embeds.device).bool()
                n_obj = len(objects_prompt_masks)
                masks_flat = objects_prompt_masks.reshape(n_obj, -1)
                hw = tile.shape[0]
                if masks_flat.shape[1] != hw:
                    side = int(hw ** 0.5)
                    if side * side != hw:
                        raise RuntimeError(f"Unexpected vit token count hw={hw} (not a square); cannot resize prompt mask")
                    m = objects_prompt_masks.float().unsqueeze(1)
                    m = F.interpolate(m, size=(side, side), mode='nearest').squeeze(1)
                    masks_flat = m.bool().reshape(n_obj, -1)

                tile_rep = tile.unsqueeze(0).repeat(n_obj, 1, 1)
                vp_parts.append(tile_rep[masks_flat])
                i_vp_img += 1
        vp_embeds = torch.cat(vp_parts, dim=0).to(device)

        # Embed tokens + inject vp embeddings into <IMG_CONTEXT> positions.
        embedding_layer = ema_model.mllm.model.language_model.get_input_embeddings()
        input_embeds = embedding_layer(input_ids.to(device))
        B, N, D = input_embeds.shape
        input_embeds_flat = input_embeds.reshape(B * N, D)
        input_ids_flat = input_ids.reshape(B * N)
        selected = (input_ids_flat == img_context_id)
        n_selected = int(selected.sum().item())
        if n_selected != int(vp_embeds.shape[0]):
            raise RuntimeError(
                f"EMA generate IMG_CONTEXT token mismatch: selected={n_selected} vp_embeds={int(vp_embeds.shape[0])}"
            )
        input_embeds_flat[selected] = vp_embeds.to(input_embeds_flat.dtype)
        input_embeds = input_embeds_flat.reshape(B, N, D)

        # Call HF generate directly (no InternVL wrapper).
        outputs = ema_model.mllm.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=gen_config,
            use_cache=True,
        )

        # IMPORTANT: Different generation wrappers return different shapes:
        # - Some return the full sequence: [prompt, generated]
        # - Some return only the generated continuation.
        #
        # We detect this by length. If `outputs` is longer than the prompt, slice off the prompt;
        # otherwise treat it as "generated only".
        prompt_len = int(input_ids.shape[1])
        if outputs.shape[1] > prompt_len:
            generated = outputs[:, prompt_len:]
            returned_full_sequence = True
        else:
            generated = outputs
            returned_full_sequence = False

        # One-time debug print (rank0 only) to validate generate() return shape behavior.
        if os.environ.get("PSEUDO_GUMBEL_DEBUG_GENERATE", "0") == "1":
            if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                pre_pad_len = int(generated.shape[1])
                sample_ids = generated[0].tolist()[: min(32, pre_pad_len)]
                try:
                    sample_text = tokenizer.decode(sample_ids, skip_special_tokens=False)
                except Exception:
                    sample_text = "<decode_failed>"
                print(
                    f"[DBG][ema.generate] prompt_len={prompt_len} outputs_len={int(outputs.shape[1])} "
                    f"returned_full={returned_full_sequence} pre_pad_len={pre_pad_len} "
                    f"pad_token_id={pad_token_id} eos_token_id={tokenizer.eos_token_id} "
                    f"sample_ids={sample_ids} sample_text={sample_text!r}",
                    flush=True,
                )

        # Pad/truncate to max_caption_len
        if generated.shape[1] < max_caption_len:
            pad_len = max_caption_len - generated.shape[1]
            generated = F.pad(generated, (0, pad_len), value=pad_token_id)
        else:
            generated = generated[:, :max_caption_len]

        # Hard guard: teacher tokens must not be entirely padding (catastrophic failure).
        if (generated != pad_token_id).sum().item() == 0:
            raise RuntimeError(
                "EMA generated tokens are entirely padding; check generate() return shape and prompt slicing."
            )

    return generated  # [B, max_caption_len]


def random_mask_tokens(tokens, mask_ratio, vocab_size, pad_token_id, device, forbidden_token_ids=None):
    """
    Step 2: Randomly mask tokens with mask_ratio.

    Args:
        tokens: [B, T] token IDs
        mask_ratio: ratio of tokens to mask
        vocab_size: vocabulary size
        pad_token_id: padding token ID
        device: device

    Returns:
        masked_tokens: [B, T] masked token IDs
    """
    masked = tokens.clone()
    batch_size, seq_len = tokens.shape

    # Create mask (don't mask padding)
    mask = torch.rand(batch_size, seq_len, device=device) < mask_ratio
    mask = mask & (tokens != pad_token_id)

    # Replace with random tokens
    random_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Avoid sampling special tokens that break <IMG_CONTEXT> embedding injection.
    if forbidden_token_ids:
        forbidden = torch.tensor(
            sorted(set(int(x) for x in forbidden_token_ids)),
            device=device,
            dtype=random_tokens.dtype,
        )
        if forbidden.numel() > 0:
            max_tries = 10
            for _ in range(max_tries):
                bad = mask & (random_tokens.unsqueeze(-1) == forbidden.view(1, 1, -1)).any(dim=-1)
                if not bool(bad.any().item()):
                    break
                random_tokens[bad] = torch.randint(
                    0, vocab_size, (int(bad.sum().item()),),
                    device=device, dtype=random_tokens.dtype,
                )
    masked[mask] = random_tokens[mask]

    return masked


def forward_for_logits(model, pixel_values, prompt_masks, masked_pseudo_toks, tokenizer, max_caption_len, device):
    """
    Step 3: Trainable model produces logits from image + mask + masked pseudo tokens.

    This is NOT a standard generation - we need to get logits for all positions.

    Args:
        model: trainable model (NOT EMA)
        pixel_values: [B, 3, 448, 448] - ImageNet normalized for InternVL
        prompt_masks: [B, 16, 16] - Visual prompt masks [0, 1]
        masked_pseudo_toks: [B, max_caption_len] masked token IDs
        tokenizer: tokenizer
        max_caption_len: max caption length
        device: device

    Returns:
        logits: [B, max_caption_len, V] logits for caption positions
    """
    batch_size = pixel_values.shape[0]

    # Images and masks are already in the correct format from dataset
    images_448 = pixel_values  # Already (B, 3, 448, 448) normalized
    # prompt_masks already (B, 16, 16)

    # Build input prompt + masked pseudo tokens as assistant "output" (teacher forcing).
    # Prompt must match DescribeAnythingReferringDataset (sa2va_4b.py) + phi3_chat template.
    IMG_CONTEXT = '<IMG_CONTEXT>'
    NUM_IMG_TOKENS = 256

    input_ids_list = []
    labels_list = []
    user_lens = []

    for i in range(batch_size):
        K = _count_vp_tokens_from_prompt_mask(
            prompt_mask=prompt_masks[i],
            num_img_tokens=NUM_IMG_TOKENS,
            device=device,
        )
        img_str = f'<img>{IMG_CONTEXT * NUM_IMG_TOKENS}</img>'
        human_input = (
            f"{img_str}\n"
            f"There are 1 part regions in the picture: region1<vp>{IMG_CONTEXT * K}</vp>.\n"
            "Please generate a detailed description for the given image region."
        )
        user_prompt = f"<|user|>\n{human_input}<|end|>\n<|assistant|>\n"
        user_ids = tokenizer.encode(user_prompt, add_special_tokens=True)
        user_lens.append(len(user_ids))

        # Target: masked pseudo tokens + template suffix "<|end|>"
        target_ids = masked_pseudo_toks[i].tolist()
        target_ids += tokenizer.encode("<|end|>", add_special_tokens=False)

        # Full sequence
        full_ids = user_ids + target_ids
        # Labels: ignore user prompt, only supervise target
        full_labels = [-100] * len(user_ids) + target_ids

        input_ids_list.append(full_ids)
        labels_list.append(full_labels)

    # Pad
    max_len = max(len(ids) for ids in input_ids_list)
    max_len = _global_max_len(max_len, device=device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, :len(ids)] = torch.tensor(ids, device=device)
        labels[i, :len(labs)] = torch.tensor(labs, device=device)
        attention_mask[i, :len(ids)] = True

    position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # -------------------------------------------------------------------------
    # IMPORTANT: Do NOT call InternVL forward here.
    #
    # InternVL `_llm_forward` requires strict equality:
    #   (#<IMG_CONTEXT> tokens in input_ids) == (len(vp_embeds))
    # and its mismatch handling can crash.
    #
    # Instead, we:
    # 1) Build base token embeddings from the LLM embedding layer.
    # 2) Compute vit/vp embeddings with the frozen vision encoder.
    # 3) Manually inject those embeddings into <IMG_CONTEXT> positions.
    # 4) Run the underlying transformer + lm_head to obtain caption logits.
    # -------------------------------------------------------------------------

    img_context_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    if img_context_id is None or img_context_id < 0:
        raise RuntimeError("Tokenizer missing <IMG_CONTEXT> token id")

    # Prepare pixel_values list in the same shape semantics as InternVL:
    # - each element is (T,3,H,W) or (3,H,W); InternVL will unsqueeze 3D to 4D.
    pixel_values_list = [images_448[i] for i in range(batch_size)]

    # Extract vit embeddings (frozen vision encoder; no grad needed)
    vision_dtype = model.mllm.model.vision_model.dtype
    with torch.no_grad():
        pv = [x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values_list]
        concat_images = torch.cat([x.to(vision_dtype) for x in pv], dim=0).to(device)
        image_flags = (torch.sum(concat_images, dim=(1, 2, 3)) != 0).long()
        if (image_flags == 0).any():
            raise RuntimeError(
                "Found all-zero image tile(s) after preprocessing (image_flags==0). "
                "This breaks vp embedding alignment; please inspect dataset preprocessing."
            )
        vit_embeds = model.mllm.model.extract_feature(concat_images)
        vit_embeds = vit_embeds[image_flags == 1]

    # Build vp_embeds exactly like InternVL.generate does (but without any model-code mutation).
    # NOTE: our training data uses 1 tile per sample, so len(vit_embeds) == batch_size.
    # Still, keep the generic shape for safety.
    prompt_masks_list = [prompt_masks[i:i + 1].to(device) for i in range(batch_size)]
    if len(vit_embeds) != batch_size:
        raise RuntimeError(
            f"Unexpected vit batch size after extract_feature: {len(vit_embeds)} (expected {batch_size}). "
            "This likely indicates multi-tile inputs; update training prompt construction to match tiling."
        )
    vp_overall_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    C = vit_embeds.shape[-1]
    vp_embeds_parts = []
    i_vp_img = 0
    for i_img in range(len(vit_embeds)):
        tile = vit_embeds[i_img].reshape(-1, C)  # (hw, C)
        vp_embeds_parts.append(tile)
        if bool(vp_overall_mask[i_img].item()):
            # One region per sample for our datasets: prompt_masks_list[i_vp_img] is (1,G,G)
            objects_prompt_masks = prompt_masks_list[i_vp_img].to(vit_embeds.device).bool()
            n_obj = len(objects_prompt_masks)
            masks_flat = objects_prompt_masks.reshape(n_obj, -1)
            # If mask grid doesn't match hw, resize it to sqrt(hw) x sqrt(hw) (nearest) before flatten.
            hw = tile.shape[0]
            if masks_flat.shape[1] != hw:
                side = int(hw ** 0.5)
                if side * side != hw:
                    raise RuntimeError(f"Unexpected vit token count hw={hw} (not a square); cannot resize prompt mask")
                m = objects_prompt_masks.float().unsqueeze(1)  # (n_obj,1,G,G)
                m = F.interpolate(m, size=(side, side), mode='nearest').squeeze(1)
                masks_flat = m.bool().reshape(n_obj, -1)

            tile_rep = tile.unsqueeze(0).repeat(n_obj, 1, 1)  # (n_obj, hw, C)
            vp_embeds_parts.append(tile_rep[masks_flat])
            i_vp_img += 1

    vp_embeds = torch.cat(vp_embeds_parts, dim=0).to(device)  # (n_vp_tokens, C)

    # Build base embeddings from token ids
    embedding_layer = model.mllm.model.language_model.get_input_embeddings()
    input_embeds = embedding_layer(input_ids)

    # Inject vp embeddings into <IMG_CONTEXT> token positions
    B, N, D = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, D)
    input_ids_flat = input_ids.reshape(B * N)
    selected = (input_ids_flat == img_context_id)
    n_selected = int(selected.sum().item())
    if n_selected != int(vp_embeds.shape[0]):
        raise RuntimeError(
            f"IMG_CONTEXT token mismatch: selected={n_selected} vp_embeds={int(vp_embeds.shape[0])}. "
            f"This indicates prompt construction is inconsistent with vp embedding construction."
        )

    # Avoid in-place writes on a view that can be treated as a leaf by autograd.
    input_embeds_flat = input_embeds_flat.clone()
    input_embeds_flat[selected] = vp_embeds.to(input_embeds_flat.dtype)
    input_embeds = input_embeds_flat.reshape(B, N, D)

    # Run underlying transformer to get hidden states, then apply lm_head only on caption positions.
    llm = model.mllm.model.language_model
    base = getattr(llm, "base_model", None)
    causal_lm = base.model if (base is not None and hasattr(base, "model")) else llm
    transformer = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    if transformer is None or lm_head is None:
        raise RuntimeError("Unexpected LLM wrapper: cannot access transformer/lm_head for logits")

    outputs = transformer(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )
    if getattr(outputs, "hidden_states", None) is not None:
        hidden_states = outputs.hidden_states[-1]
    elif isinstance(outputs, (tuple, list)) and len(outputs) >= 3 and outputs[2] is not None:
        hidden_states = outputs[2][-1]
    else:
        hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state

    # Extract logits aligned to caption tokens.
    # For a causal LM, logits at position t predict the token at position t+1.
    # We want distributions for caption tokens token_0..token_{T-1} where:
    #   token_0 is predicted from the last prompt token (index P-1),
    #   token_1 is predicted from token_0 (index P), ...
    # Therefore we take logits indices [P-1 : P-1+T] for each sample (P may differ per sample).
    vocab_size = int(lm_head.out_features) if hasattr(lm_head, "out_features") else int(lm_head.weight.shape[0])
    # NOTE: full-vocab logits are extremely large; keep them in the model dtype to reduce memory.
    caption_logits = torch.zeros(
        batch_size, max_caption_len, vocab_size,
        device=hidden_states.device, dtype=hidden_states.dtype
    )
    for i, p_len in enumerate(user_lens):
        start = max(p_len - 1, 0)
        end = start + max_caption_len
        h = hidden_states[i, start:end, :]  # (T, D)
        caption_logits[i] = lm_head(h).to(dtype=caption_logits.dtype)

    return caption_logits


def topk_gumbel_softmax(logits, tau, topk, embedding_layer):
    """
    Step 4: Top-k sparse Gumbel-Softmax with Straight-Through.

    Args:
        logits: [B, T, V] full vocabulary logits
        tau: temperature
        topk: k for top-k
        embedding_layer: embedding layer for getting embeddings

    Returns:
        text_embeds: [B, T, D] differentiable text embeddings
    """
    B, T, V = logits.shape

    # Get top-k logits and indices
    topk_vals, topk_idx = logits.topk(topk, dim=-1)  # [B, T, k]

    # Gumbel-Softmax on top-k
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(topk_vals) + 1e-10) + 1e-10)
    y_soft = F.softmax((topk_vals + gumbel_noise) / tau, dim=-1)  # [B, T, k]

    # Straight-through: hard forward, soft backward
    index = y_soft.argmax(dim=-1, keepdim=True)  # [B, T, 1]
    y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
    y = y_hard - y_soft.detach() + y_soft  # [B, T, k] with ST gradient

    # Memory note:
    # Materializing `E_topk = E[topk_idx]` creates a huge [B,T,k,D] tensor.
    # For typical settings (B=16,T=256,k=512,D=2048) this can OOM.
    #
    # We compute the weighted sum in chunks over k to keep peak memory bounded.
    E = embedding_layer.weight  # [V, D]
    B, T, K = topk_idx.shape
    D = int(E.shape[1])
    text_embeds = torch.zeros((B, T, D), device=logits.device, dtype=E.dtype)

    # Chunk size can be tuned; smaller chunks reduce peak memory of [B,T,chunk,D] gathers.
    chunk_k = int(os.environ.get("PSEUDO_GUMBEL_TOPK_CHUNK", "8"))
    if chunk_k <= 0:
        chunk_k = K

    for s in range(0, K, chunk_k):
        e = min(s + chunk_k, K)
        idx_chunk = topk_idx[:, :, s:e]  # [B,T,c]
        y_chunk = y[:, :, s:e].to(E.dtype)  # [B,T,c]
        emb_chunk = F.embedding(idx_chunk, E)  # [B,T,c,D]
        text_embeds = text_embeds + (y_chunk.unsqueeze(-1) * emb_chunk).sum(dim=-2)

    return text_embeds


def forward_mask_with_text_embeds(
    model,
    pixel_values,
    g_pixel_values,
    text_embeds,
    gt_masks,
    seg_token_id,
    img_context_id,
    tokenizer,
    device,
):
    """
    Step 5: Trainable model predicts mask from image + text_embeds.

    CRITICAL: Use inputs_embeds, not input_ids!

    Args:
        model: trainable model
        pixel_values: [B, 3, 448, 448] - ImageNet normalized for InternVL
        g_pixel_values: [B, 3, 1024, 1024] - [0, 255] uint8 for SAM2
        text_embeds: [B, T, D] differentiable text embeddings
        gt_masks: [B, 1024, 1024] - ground truth masks [0, 1]
        seg_token_id: [SEG] token ID
        img_context_id: <IMG_CONTEXT> token ID
        tokenizer: tokenizer
        device: device

    Returns:
        loss_dict:
          - 'loss': segmentation loss tensor (mask + dice)
          - 'llm_loss_t': LM loss tensor on assistant answer (keeps `[SEG]` generation)
          - 'mask_loss'/'dice_loss'/'llm_loss': python floats for logging
    """
    batch_size = pixel_values.shape[0]

    # Images are already in the correct format from dataset
    images_448 = pixel_values  # Already (B, 3, 448, 448) normalized

    # Build prompt template matching ReferSegmDataset (RefCOCO_Dataset.py) used by sa2va_4b.py.
    #
    # User input starts with "<image>\n" + SEG_QUESTIONS.format(class_name=phrase).
    # During encoding, "<image>" is replaced with "<img>{IMG_CONTEXT*256}</img>".
    # Importantly, user prompt DOES NOT contain "[SEG]"; assistant output contains "[SEG]".
    #
    # ReferSegmDataset also overrides phi3_chat INSTRUCTION to:
    #   "<|user|>\n{input}<|end|><|assistant|>\n"
    IMG_CONTEXT = '<IMG_CONTEXT>'
    NUM_IMG_TOKENS = 256

    from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST

    question_template = random.choice(SEG_QUESTIONS)
    if "{class_name}" not in question_template:
        raise ValueError(f"Unexpected SEG_QUESTIONS template: {question_template}")
    q_prefix, q_suffix = question_template.split("{class_name}")

    answer_template = random.choice(ANSWER_LIST)
    if answer_template.count("[SEG]") != 1:
        raise ValueError(f"Unexpected ANSWER_LIST template: {answer_template}")
    a_prefix, a_suffix = answer_template.split("[SEG]")

    # Match ReferSegmDataset:
    # - human text begins with "<image>\\n" + question
    # - then "<image>" is replaced with "<img>{IMG_CONTEXT*num_img_tokens}</img>"
    img_str = f"<img>{IMG_CONTEXT * NUM_IMG_TOKENS}</img>"
    human_prefix = f"<image>\n{q_prefix}".replace("<image>", img_str)
    user_prefix_str = f"<|user|>\n{human_prefix}"
    user_suffix_str = f"{q_suffix}<|end|><|assistant|>\n"

    prefix_ids = tokenizer.encode(user_prefix_str, add_special_tokens=True)
    suffix_ids = tokenizer.encode(user_suffix_str, add_special_tokens=False)

    assistant_ids = (
        tokenizer.encode(a_prefix, add_special_tokens=False)
        + [seg_token_id]
        + tokenizer.encode(a_suffix, add_special_tokens=False)
        + tokenizer.encode("<|end|>", add_special_tokens=False)
    )

    caption_len = text_embeds.shape[1]  # T
    total_len = len(prefix_ids) + caption_len + len(suffix_ids) + len(assistant_ids)

    # Build input_ids (placeholders for text_embeds positions)
    input_ids = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for i in range(batch_size):
        input_ids[i, :len(prefix_ids)] = torch.tensor(prefix_ids, device=device)
        # Placeholder for text_embeds (will be replaced by embeddings)
        input_ids[i, len(prefix_ids):len(prefix_ids)+caption_len] = pad_token_id
        offset = len(prefix_ids) + caption_len
        input_ids[i, offset:offset + len(suffix_ids)] = torch.tensor(suffix_ids, device=device)
        input_ids[i, offset + len(suffix_ids):] = torch.tensor(assistant_ids, device=device)

    attention_mask = torch.ones(batch_size, total_len, dtype=torch.bool, device=device)
    position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # LM labels: only supervise the assistant output (including `[SEG]`), ignore everything else.
    labels = torch.full((batch_size, total_len), -100, dtype=torch.long, device=device)
    assistant_start = total_len - len(assistant_ids)
    labels[:, assistant_start:total_len] = torch.tensor(assistant_ids, device=device, dtype=torch.long).unsqueeze(0)

    # Make seq-len consistent across ranks to reduce per-rank VRAM skew (DDP).
    global_total_len = _global_max_len(total_len, device=device)
    if global_total_len > total_len:
        input_ids = _pad_2d_right(input_ids, global_total_len, pad_token_id)
        attention_mask = _pad_2d_right(attention_mask.to(torch.long), global_total_len, 0).to(torch.bool)
        position_ids = torch.arange(global_total_len, device=device).unsqueeze(0).expand(batch_size, -1)
        labels = _pad_2d_right(labels, global_total_len, -100)

    # Get base embeddings from embedding layer
    embedding_layer = model.mllm.model.language_model.get_input_embeddings()
    base_embeds = embedding_layer(input_ids)  # [B, total_len, D]

    # Replace text_embeds positions
    soft_start = len(prefix_ids)
    soft_end = soft_start + caption_len
    input_embeds = base_embeds.clone()
    input_embeds[:, soft_start:soft_end, :] = text_embeds

    # Replace image token positions with vision embeddings
    concat_images = torch.stack([images_448[i] for i in range(batch_size)], dim=0)
    vit_embeds = model.mllm.model.extract_feature(concat_images)
    vit_embeds = vit_embeds.to(input_embeds.dtype)

    # Find <IMG_CONTEXT> positions and replace with vision embeddings
    B, N, C = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)
    selected = (input_ids_flat == img_context_id)

    n_selected = selected.sum()
    vit_flat = vit_embeds.reshape(-1, C)
    if n_selected > len(vit_flat):
        # Need to expand vit_embeds
        expand = n_selected // len(vit_flat) + 1
        vit_flat = torch.cat([vit_flat] * expand, dim=0)
    input_embeds_flat[selected] = vit_flat[:n_selected]
    input_embeds = input_embeds_flat.reshape(B, N, C)

    # Forward through LLM *without* materializing full-vocab logits.
    #
    # For segmentation, we only need last-layer hidden states (to extract [SEG] embeddings).
    # For LM supervision on the fixed assistant answer, we only need logits for supervised positions,
    # not the full [B, L, V] tensor (which can OOM for large bs).
    llm = model.mllm.model.language_model
    base = getattr(llm, "base_model", None)
    causal_lm = base.model if (base is not None and hasattr(base, "model")) else llm
    transformer = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)

    if transformer is None or lm_head is None:
        outputs = llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,
        )
        if getattr(outputs, "loss", None) is not None and getattr(outputs, "hidden_states", None) is not None:
            llm_loss = outputs.loss
            hidden_states = outputs.hidden_states[-1]
        elif isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
            llm_loss = outputs[0]
            hidden_states = outputs[2][-1]
        else:
            raise RuntimeError("Unexpected LLM output structure in fallback branch")
    else:
        outputs = transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        if getattr(outputs, "hidden_states", None) is not None:
            hidden_states = outputs.hidden_states[-1]
        elif isinstance(outputs, (tuple, list)) and len(outputs) >= 3 and outputs[2] is not None:
            hidden_states = outputs[2][-1]
        else:
            hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state

        shift_labels = labels[:, 1:].contiguous()
        valid = shift_labels != -100
        if valid.any():
            h = hidden_states[:, :-1, :][valid]
            logits_small = lm_head(h).float()
            llm_loss = F.cross_entropy(
                logits_small, shift_labels[valid], ignore_index=-100, reduction="mean"
            )
        else:
            llm_loss = torch.tensor(0.0, device=device)
    hidden_states_transformed = model.text_hidden_fcs(hidden_states)

    # Find [SEG] positions
    seg_mask = input_ids == seg_token_id
    seg_counts = seg_mask.int().sum(-1)

    if seg_counts.sum() == 0:
        print(f"⚠ Warning: No [SEG] tokens found in forward_mask_with_text_embeds!")
        return {
            'loss': torch.tensor(0.0, device=device, requires_grad=True),
            'mask_loss': 0.0,
            'dice_loss': 0.0,
            'llm_loss': 0.0,
        }

    pred_embeddings = hidden_states_transformed[seg_mask]

    # SAM2 forward for mask prediction
    g_pixel_values = g_pixel_values.to(device)
    g_pixel_values = torch.stack(
        [
            model.grounding_encoder.preprocess_image(g_pixel_values[i])
            for i in range(batch_size)
        ],
        dim=0,
    )

    num_objs = 1
    sam_states = model.grounding_encoder.get_sam2_embeddings(
        g_pixel_values, expand_size=num_objs
    )

    # Prepare language embeddings
    pred_list = torch.split(pred_embeddings, seg_counts.tolist(), dim=0)
    pred_list = [item for item in pred_list if len(item) > 0]

    if len(pred_list) == 0:
        return {
            'loss': torch.tensor(0.0, device=device, requires_grad=True),
            'mask_loss': 0.0,
            'dice_loss': 0.0,
            'llm_loss': 0.0,
        }

    lang_embeds = torch.stack([emb[0] for emb in pred_list], dim=0)[:, None]

    # Predict masks
    pred_masks = model.grounding_encoder.inject_language_embd(
        sam_states, lang_embeds, nf_nobj=(batch_size, num_objs)
    )

    # Compute loss (align with Sa2VA: use model.loss_* and optional point sampling)
    pred_size = pred_masks.shape[-2:]
    gt_resized = F.interpolate(
        gt_masks.unsqueeze(1).float(), size=pred_size, mode='nearest'
    )  # (B, 1, H, W)
    gt_resized = gt_resized.expand(-1, num_objs, -1, -1)  # (B, num_objs, H, W)

    pred_flat = pred_masks.flatten(0, 1)  # (B*num_objs, H, W)
    gt_flat = gt_resized.flatten(0, 1)  # (B*num_objs, H, W)

    if getattr(model, "loss_sample_points", False):
        sampled_pred_mask, sampled_gt_mask = model.sample_points(pred_flat, gt_flat)
        loss_dice = model.loss_dice(
            sampled_pred_mask,
            sampled_gt_mask,
            avg_factor=(len(gt_flat) + 1e-4),
        )
        loss_mask = model.loss_mask(
            sampled_pred_mask.reshape(-1),
            sampled_gt_mask.reshape(-1),
            avg_factor=(pred_flat.shape[0] * sampled_pred_mask.shape[1] + 1e-4),
        )
    else:
        loss_mask = model.loss_mask(pred_flat, gt_flat)
        loss_dice = model.loss_dice(pred_flat, gt_flat)

    total_loss = loss_mask + loss_dice

    return {
        'loss': total_loss,
        'mask_loss': loss_mask.item() if torch.is_tensor(loss_mask) else float(loss_mask),
        'dice_loss': loss_dice.item() if torch.is_tensor(loss_dice) else float(loss_dice),
        'llm_loss_t': llm_loss if torch.is_tensor(llm_loss) else torch.tensor(float(llm_loss), device=device),
        'llm_loss': llm_loss.item() if torch.is_tensor(llm_loss) else float(llm_loss),
    }


def predict_masks_with_text_embeds(
    model,
    pixel_values,
    g_pixel_values,
    text_embeds,
    seg_token_id,
    tokenizer,
    device,
):
    """
    Predict segmentation masks from image + text_embeds via the standard Sa2VA `[SEG]` mechanism.

    This mirrors `forward_mask_with_text_embeds` but returns `pred_masks` and does not compute any loss.
    """
    batch_size = pixel_values.shape[0]

    IMG_CONTEXT = '<IMG_CONTEXT>'
    NUM_IMG_TOKENS = 256

    from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST

    question_template = random.choice(SEG_QUESTIONS)
    if "{class_name}" not in question_template:
        raise ValueError(f"Unexpected SEG_QUESTIONS template: {question_template}")
    q_prefix, q_suffix = question_template.split("{class_name}")

    answer_template = random.choice(ANSWER_LIST)
    if answer_template.count("[SEG]") != 1:
        raise ValueError(f"Unexpected ANSWER_LIST template: {answer_template}")
    a_prefix, a_suffix = answer_template.split("[SEG]")

    img_str = f"<img>{IMG_CONTEXT * NUM_IMG_TOKENS}</img>"
    human_prefix = f"<image>\n{q_prefix}".replace("<image>", img_str)
    user_prefix_str = f"<|user|>\n{human_prefix}"
    user_suffix_str = f"{q_suffix}<|end|><|assistant|>\n"

    prefix_ids = tokenizer.encode(user_prefix_str, add_special_tokens=True)
    suffix_ids = tokenizer.encode(user_suffix_str, add_special_tokens=False)
    assistant_ids = (
        tokenizer.encode(a_prefix, add_special_tokens=False)
        + [seg_token_id]
        + tokenizer.encode(a_suffix, add_special_tokens=False)
        + tokenizer.encode("<|end|>", add_special_tokens=False)
    )

    caption_len = int(text_embeds.shape[1])
    total_len = len(prefix_ids) + caption_len + len(suffix_ids) + len(assistant_ids)
    total_len = _global_max_len(total_len, device=device)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((batch_size, total_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
    labels = torch.full((batch_size, total_len), -100, dtype=torch.long, device=device)

    prefix_len = len(prefix_ids)
    suffix_len = len(suffix_ids)
    assistant_len = len(assistant_ids)

    input_ids[:, :prefix_len] = torch.tensor(prefix_ids, device=device).unsqueeze(0)
    input_ids[:, prefix_len + caption_len: prefix_len + caption_len + suffix_len] = torch.tensor(
        suffix_ids, device=device
    ).unsqueeze(0)
    input_ids[:, prefix_len + caption_len + suffix_len: prefix_len + caption_len + suffix_len + assistant_len] = (
        torch.tensor(assistant_ids, device=device).unsqueeze(0)
    )

    # Only supervise assistant answer (keeps `[SEG]` embedding extraction stable).
    labels[:, prefix_len + caption_len + suffix_len: prefix_len + caption_len + suffix_len + assistant_len] = (
        torch.tensor(assistant_ids, device=device).unsqueeze(0)
    )

    attention_mask[:, : prefix_len + caption_len + suffix_len + assistant_len] = True
    position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Build base embeddings and inject image/vp embeddings into <IMG_CONTEXT> slots.
    img_context_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    if img_context_id is None or img_context_id < 0:
        raise RuntimeError("Tokenizer missing <IMG_CONTEXT> token id")

    # Build input embeddings: prefix + provided text_embeds + suffix + assistant ids.
    embedding_layer = model.mllm.model.language_model.get_input_embeddings()
    base_embeds = embedding_layer(input_ids)
    input_embeds = base_embeds.clone()
    input_embeds[:, prefix_len: prefix_len + caption_len, :] = text_embeds.to(input_embeds.dtype)

    # Replace image token positions with vision embeddings (no vp tokens in RefCOCO-style prompt).
    vision_dtype = model.mllm.model.vision_model.dtype
    with torch.no_grad():
        concat_images = torch.stack([pixel_values[i] for i in range(batch_size)], dim=0).to(device=device, dtype=vision_dtype)
        vit_embeds = model.mllm.model.extract_feature(concat_images).to(dtype=input_embeds.dtype)

    B, N, D = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, D)
    input_ids_flat = input_ids.reshape(B * N)
    selected = (input_ids_flat == img_context_id)
    n_selected = int(selected.sum().item())
    vit_flat = vit_embeds.reshape(-1, D)
    if n_selected > int(vit_flat.shape[0]):
        expand = n_selected // int(vit_flat.shape[0]) + 1
        vit_flat = torch.cat([vit_flat] * expand, dim=0)
    input_embeds_flat = input_embeds_flat.clone()
    input_embeds_flat[selected] = vit_flat[:n_selected]
    input_embeds = input_embeds_flat.reshape(B, N, D)

    # Run LLM transformer to get hidden states.
    llm = model.mllm.model.language_model
    base = getattr(llm, "base_model", None)
    causal_lm = base.model if (base is not None and hasattr(base, "model")) else llm
    transformer = getattr(causal_lm, "model", None)
    if transformer is None:
        outputs = llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
    else:
        outputs = transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) is not None else outputs[0]

    hidden_states_transformed = model.text_hidden_fcs(hidden_states)

    seg_mask = input_ids == seg_token_id
    seg_counts = seg_mask.int().sum(-1)
    if seg_counts.sum() == 0:
        raise RuntimeError("No [SEG] token found while predicting mask")
    pred_embeddings = hidden_states_transformed[seg_mask]

    # SAM2 forward for mask prediction
    g_pixel_values = g_pixel_values.to(device)
    g_pixel_values = torch.stack(
        [model.grounding_encoder.preprocess_image(g_pixel_values[i]) for i in range(batch_size)],
        dim=0,
    )
    num_objs = 1
    sam_states = model.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)

    pred_list = torch.split(pred_embeddings, seg_counts.tolist(), dim=0)
    pred_list = [item for item in pred_list if len(item) > 0]
    if not pred_list:
        raise RuntimeError("Empty [SEG] embeddings split")
    lang_embeds = torch.stack([emb[0] for emb in pred_list], dim=0)[:, None]
    pred_masks = model.grounding_encoder.inject_language_embd(sam_states, lang_embeds, nf_nobj=(batch_size, num_objs))
    return pred_masks


def compute_dam_caption_ce_loss(
    model,
    pixel_values,
    prompt_masks,
    captions,
    tokenizer,
    max_caption_len,
    device,
):
    """
    Teacher-forced caption CE loss on DAM captioning prompt (mask -> caption).

    `prompt_masks` is a single-region grid mask per sample (B,16,16) uint8/bool.
    """
    batch_size = pixel_values.shape[0]
    IMG_CONTEXT = '<IMG_CONTEXT>'
    NUM_IMG_TOKENS = 256

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    input_ids_list = []
    labels_list = []
    user_lens = []

    for i in range(batch_size):
        pm = prompt_masks[i]
        K = _count_vp_tokens_from_prompt_mask(pm, num_img_tokens=NUM_IMG_TOKENS, device=device)
        img_str = f'<img>{IMG_CONTEXT * NUM_IMG_TOKENS}</img>'
        human_input = (
            f"{img_str}\n"
            f"There are 1 part regions in the picture: region1<vp>{IMG_CONTEXT * K}</vp>.\n"
            "Please generate a detailed description for the given image region."
        )
        user_prompt = f"<|user|>\n{human_input}<|end|>\n<|assistant|>\n"
        user_ids = tokenizer.encode(user_prompt, add_special_tokens=True)
        user_lens.append(len(user_ids))

        cap_ids = tokenizer.encode(str(captions[i]), add_special_tokens=False)
        cap_ids = cap_ids[:max_caption_len]
        if len(cap_ids) < max_caption_len:
            cap_ids = cap_ids + [pad_token_id] * (max_caption_len - len(cap_ids))

        full_ids = user_ids + cap_ids
        full_labels = [-100] * len(user_ids) + cap_ids
        input_ids_list.append(full_ids)
        labels_list.append(full_labels)

    max_len = max(len(ids) for ids in input_ids_list)
    max_len = _global_max_len(max_len, device=device)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, :len(ids)] = torch.tensor(ids, device=device)
        labels[i, :len(labs)] = torch.tensor(labs, device=device)
        attention_mask[i, :len(ids)] = True
    position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)

    img_context_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    if img_context_id is None or img_context_id < 0:
        raise RuntimeError("Tokenizer missing <IMG_CONTEXT> token id")

    # vit/vp embeds
    vision_dtype = model.mllm.model.vision_model.dtype
    pixel_values_list = [pixel_values[i] for i in range(batch_size)]
    with torch.no_grad():
        pv = [x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values_list]
        concat_images = torch.cat([x.to(vision_dtype) for x in pv], dim=0).to(device)
        image_flags = (torch.sum(concat_images, dim=(1, 2, 3)) != 0).long()
        vit_embeds = model.mllm.model.extract_feature(concat_images)
        vit_embeds = vit_embeds[image_flags == 1]

    prompt_masks_list = [prompt_masks[i:i + 1].to(device) for i in range(batch_size)]
    if len(vit_embeds) != batch_size:
        raise RuntimeError(f"Unexpected vit batch size {len(vit_embeds)} (expected {batch_size})")
    vp_overall_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    C = vit_embeds.shape[-1]
    vp_parts = []
    i_vp_img = 0
    for i_img in range(len(vit_embeds)):
        tile = vit_embeds[i_img].reshape(-1, C)
        vp_parts.append(tile)
        if bool(vp_overall_mask[i_img].item()):
            objects_prompt_masks = prompt_masks_list[i_vp_img].to(vit_embeds.device).bool()
            n_obj = len(objects_prompt_masks)
            masks_flat = objects_prompt_masks.reshape(n_obj, -1)
            hw = tile.shape[0]
            if masks_flat.shape[1] != hw:
                side = int(hw ** 0.5)
                m = objects_prompt_masks.float().unsqueeze(1)
                m = F.interpolate(m, size=(side, side), mode='nearest').squeeze(1)
                masks_flat = m.bool().reshape(n_obj, -1)
            tile_rep = tile.unsqueeze(0).repeat(n_obj, 1, 1)
            vp_parts.append(tile_rep[masks_flat])
            i_vp_img += 1
    vp_embeds = torch.cat(vp_parts, dim=0).to(device)

    embedding_layer = model.mllm.model.language_model.get_input_embeddings()
    input_embeds = embedding_layer(input_ids)
    B, N, D = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, D)
    input_ids_flat = input_ids.reshape(B * N)
    selected = (input_ids_flat == img_context_id)
    n_selected = int(selected.sum().item())
    if n_selected != int(vp_embeds.shape[0]):
        raise RuntimeError(
            f"IMG_CONTEXT token mismatch in compute_dam_caption_ce_loss: selected={n_selected} vp={int(vp_embeds.shape[0])}"
        )
    input_embeds_flat = input_embeds_flat.clone()
    input_embeds_flat[selected] = vp_embeds.to(input_embeds_flat.dtype)
    input_embeds = input_embeds_flat.reshape(B, N, D)

    llm = model.mllm.model.language_model
    base = getattr(llm, "base_model", None)
    causal_lm = base.model if (base is not None and hasattr(base, "model")) else llm
    transformer = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    if transformer is None or lm_head is None:
        outputs = llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,
        )
        return outputs.loss

    outputs = transformer(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) is not None else outputs[0]

    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    if not valid.any():
        return torch.tensor(0.0, device=device)
    h = hidden_states[:, :-1, :][valid]
    logits_small = lm_head(h).float()
    loss = F.cross_entropy(logits_small, shift_labels[valid], ignore_index=-100, reduction="mean")
    return loss
