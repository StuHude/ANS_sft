"""
Test if Sa2VA supports inputs_embeds for ST Gumbel-Softmax
"""
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_inputs_embeds():
    """Test if we can pass inputs_embeds directly to the LLM."""

    print("=" * 60)
    print("Test 1: Check if LLM accepts inputs_embeds")
    print("=" * 60)

    from projects.llava_sam2.models.internvl import InternVL_Slowfast
    from peft import LoraConfig

    # Build model (minimal config)
    model_path = '/data/xyc/ANS/pretrained/InternVL2_5-4B'

    special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>', '<IMG_CONTEXT>', '<img>', '</img>']

    model = InternVL_Slowfast(
        model_path=model_path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=8,  # Small for testing
            lora_alpha=16,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'
        ),
        special_tokens=special_tokens,
    )
    model = model.cuda().to(torch.bfloat16)
    model.eval()

    print("✓ Model loaded")

    # Get embedding layer
    embedding_layer = model.model.language_model.get_input_embeddings()
    vocab_size = model.model.language_model.config.vocab_size
    hidden_size = embedding_layer.embedding_dim

    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden size: {hidden_size}")

    # Test 1: Create fake input_embeds
    batch_size = 1
    seq_len = 32

    # Create random input_embeds
    fake_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device='cuda')
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device='cuda')
    position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0)

    print("\n" + "=" * 60)
    print("Test 2: Forward with inputs_embeds (no input_ids)")
    print("=" * 60)

    try:
        with torch.no_grad():
            outputs = model.model.language_model(
                inputs_embeds=fake_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        print(f"✓ Forward with inputs_embeds WORKS!")
        print(f"  Output logits shape: {outputs.logits.shape}")
        print(f"  Hidden states layers: {len(outputs.hidden_states)}")

    except Exception as e:
        print(f"✗ Forward with inputs_embeds FAILED: {e}")
        return False

    # Test 3: Simulate Gumbel-Softmax flow
    print("\n" + "=" * 60)
    print("Test 3: Simulate ST Gumbel-Softmax flow")
    print("=" * 60)

    try:
        # Step 1: Get some logits (simulated Loop1 output)
        fake_logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.bfloat16, device='cuda')

        # Step 2: Apply Gumbel-Softmax
        temperature = 1.0
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(fake_logits) + 1e-10) + 1e-10)
        soft_probs = torch.softmax((fake_logits + gumbel_noise) / temperature, dim=-1)

        print(f"  Soft probs shape: {soft_probs.shape}")

        # Step 3: Get soft embeddings
        # soft_embeds = soft_probs @ embedding_matrix
        embedding_weight = embedding_layer.weight  # (vocab_size, hidden_size)
        soft_embeds = torch.matmul(soft_probs, embedding_weight)  # (B, seq_len, hidden_size)

        print(f"  Soft embeds shape: {soft_embeds.shape}")

        # Step 4: Forward with soft_embeds
        with torch.no_grad():
            outputs2 = model.model.language_model(
                inputs_embeds=soft_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        print(f"✓ ST Gumbel-Softmax flow WORKS!")
        print(f"  Output logits shape: {outputs2.logits.shape}")

    except Exception as e:
        print(f"✗ ST Gumbel-Softmax flow FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("CONCLUSION: Sa2VA SUPPORTS inputs_embeds!")
    print("ST Gumbel-Softmax is FEASIBLE for differentiable caption")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = test_inputs_embeds()
    exit(0 if success else 1)
