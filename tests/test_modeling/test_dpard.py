import unittest

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import DFlashDraftModel


class DPardForwardTest(unittest.TestCase):
    def check_forward(self, device, dtype, loss_type="dpard"):
        torch.manual_seed(42)
        config = Qwen3Config(
            architectures=["DFlashDraftModel"],
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=3,
            num_target_layers=36,
            head_dim=8,
            max_position_embeddings=128,
            vocab_size=64,
            layer_types=["full_attention"] * 3,
            dflash_config={"block_size": 16, "target_layer_ids": [1, 17, 33]},
        )
        config._attn_implementation = "sdpa"
        subject = OnlineDFlashModel(
            DFlashDraftModel(config),
            nn.Linear(32, 64, bias=False).requires_grad_(False),
            nn.Embedding(64, 32).requires_grad_(False),
            mask_token_id=63,
            block_size=16,
            num_anchors=2,
            objective_chunk_blocks=1,
            attention_backend="sdpa",
            loss_type=loss_type,
        ).to(device=device, dtype=dtype)
        ids = torch.randint(0, 63, (2, 40), device=device)
        features = torch.randn(2, 40, 96, device=device, dtype=dtype)
        teacher = torch.randn(2, 40, 32, device=device, dtype=dtype)
        mask = torch.ones_like(ids, dtype=torch.float)
        values = []
        for detailed in (False, True):
            teacher_projections = []
            hook = subject.lm_head.register_forward_pre_hook(
                lambda module, args: teacher_projections.append(args[0].shape)
                if not args[0].requires_grad else None
            )
            subject.zero_grad(set_to_none=True)
            torch.manual_seed(7)
            loss, accuracy, metrics = subject(
                ids,
                features,
                mask,
                target_last_hidden_states=teacher,
                collect_detailed_metrics=detailed,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(accuracy))
            loss.backward()
            hook.remove()
            if not detailed:
                self.assertEqual(teacher_projections, [torch.Size([2, 40, 32])])
            grads = [
                p.grad for p in subject.draft_model.parameters() if p.grad is not None
            ]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.float().abs().sum().item() for g in grads), 0)
            self.assertTrue(all(p.grad is None for p in subject.lm_head.parameters()))
            values.append(loss.detach())
        torch.testing.assert_close(*values)

    def test_b16_forward_backward(self):
        self.check_forward("cpu", torch.float32)

    def test_dpala_b16_forward_backward(self):
        self.check_forward("cpu", torch.float32, "dpala")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_b16_bf16_forward_backward(self):
        self.check_forward("cuda", torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dpala_b16_bf16_forward_backward(self):
        self.check_forward("cuda", torch.bfloat16, "dpala")


if __name__ == "__main__":
    unittest.main()
