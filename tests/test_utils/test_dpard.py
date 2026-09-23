"""DFlash1 D-PARD actor, credit, and sequence-anchor reduction."""

import math
import unittest

import torch
from torch import nn

from tests.test_utils.test_dflash_losses import OnlineDFlashModel


def model(loss_type, **kwargs):
    return OnlineDFlashModel(
        draft_model=nn.Identity(),
        target_lm_head=nn.Identity(),
        target_embed_tokens=nn.Embedding(2, 2),
        mask_token_id=0,
        block_size=3,
        attention_backend="sdpa",
        loss_type=loss_type,
        **kwargs,
    )


class DPardTests(unittest.TestCase):
    def setUp(self):
        self.logits = (
            torch.tensor([math.log(0.25), math.log(0.75)])
            .expand(2, 2, 3, 2)
            .clone()
            .requires_grad_()
        )
        self.teacher = torch.zeros_like(self.logits, requires_grad=True)
        self.labels = torch.zeros(2, 2, 3, dtype=torch.long)
        self.mask = torch.tensor(
            [[[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]
        )

    def terms(self, subject, *, teacher=None, sl=slice(None)):
        scale = subject._sequence_anchor_scale(self.mask)
        return subject._dflash_objective_chunk_terms(
            self.logits[:, sl],
            self.labels[:, sl],
            self.mask[:, sl],
            self.labels[:, sl],
            scale[:, sl],
            None if teacher is None else teacher[:, sl],
        )

    def test_exact_actor_credit_and_stopped_teacher(self):
        subject = model("dpard")
        terms = self.terms(subject, teacher=self.teacher)
        renyi = -2 * math.log(math.sqrt(0.125) + math.sqrt(0.375))
        torch.testing.assert_close(terms.loss_den, torch.tensor(2.0))
        torch.testing.assert_close(terms.ce_loss_num, torch.tensor(renyi * 3.28125))
        (terms.ce_loss_num / terms.loss_den).backward()
        self.assertIsNone(self.teacher.grad)
        gradient = torch.tensor([0.25, 0.75]) - torch.tensor(
            [math.sqrt(0.125), math.sqrt(0.375)]
        ) / (math.sqrt(0.125) + math.sqrt(0.375))
        torch.testing.assert_close(self.logits.grad[0, 0, 1], gradient * 0.41015625)
        torch.testing.assert_close(self.logits.grad[1, 0, 1], gradient * 0.4375)

    def test_chunking_preserves_global_anchor_scale(self):
        subject = model("dpard")
        full = self.terms(subject, teacher=self.teacher)
        chunks = [
            self.terms(subject, teacher=self.teacher, sl=slice(i, i + 1))
            for i in range(2)
        ]
        torch.testing.assert_close(full.ce_loss_num, sum(x.ce_loss_num for x in chunks))
        torch.testing.assert_close(full.loss_den, sum(x.loss_den for x in chunks))

    def test_point_mass_target_matches_dpace(self):
        teacher = torch.tensor([0.0, -float("inf")]).expand_as(self.logits)
        rd = self.terms(model("dpard"), teacher=teacher)
        ce = self.terms(model("dpace"))
        torch.testing.assert_close(rd.ce_loss_num, ce.ce_loss_num)
        torch.testing.assert_close(rd.loss_den, ce.loss_den)
        overlap = self.terms(model("dpala"), teacher=teacher)
        torch.testing.assert_close(overlap.ce_loss_num, ce.ce_loss_num)
        torch.testing.assert_close(overlap.loss_den, ce.loss_den)

    def test_overlap_objectives_reject_selector_and_lk(self):
        for objective in ("dpard", "dpala"):
            with self.subTest(objective=objective):
                with self.assertRaisesRegex(ValueError, "LK"):
                    model(objective, lk_loss_type="tv")
                draft = nn.Identity()
                draft.candidate_selector = nn.Linear(2, 2)
                with self.assertRaisesRegex(ValueError, "DFlash1"):
                    OnlineDFlashModel(draft, nn.Identity(), nn.Embedding(2, 2),
                                      mask_token_id=0, loss_type=objective)

    def test_static_ce_anchor_control_is_explicit(self):
        anchor = self.terms(model("dflash", normalize_by_anchors=True))
        original = self.terms(model("dflash"))
        self.assertAlmostEqual(
            (anchor.ce_loss_num / anchor.loss_den).item(),
            -math.log(0.25) * 1.5,
            places=5,
        )
        self.assertAlmostEqual(
            (original.ce_loss_num / original.loss_den).item(), -math.log(0.25), places=5
        )

    def test_missing_teacher_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "target_last_hidden_states"):
            self.terms(model("dpard"))

    def test_dpala_actor_and_detached_overlap_credit(self):
        terms = self.terms(model("dpala"), teacher=self.teacher)
        torch.testing.assert_close(terms.loss_den, torch.tensor(2.0))
        torch.testing.assert_close(
            terms.ce_loss_num, torch.tensor(-math.log(0.75) * 3.28125)
        )
        (terms.ce_loss_num / terms.loss_den).backward()
        self.assertIsNone(self.teacher.grad)
        torch.testing.assert_close(
            self.logits.grad[0, 0, 1], torch.tensor([-0.25, 0.25]) * 0.41015625
        )

    def test_cached_teacher_logits_preserve_loss_and_gradient(self):
        subject = model("dpard")
        original = self.terms(subject, teacher=self.teacher)
        expected_grad = torch.autograd.grad(original.ce_loss_num, self.logits)[0]
        positions = torch.arange(3).expand(2, 2, 3)
        cached = subject._dflash_objective_chunk_terms(
            self.logits, self.labels, self.mask, self.labels,
            subject._sequence_anchor_scale(self.mask), None, positions,
            target_logits=self.teacher[:, 0].detach(),
        )
        torch.testing.assert_close(original.ce_loss_num, cached.ce_loss_num)
        torch.testing.assert_close(
            expected_grad, torch.autograd.grad(cached.ce_loss_num, self.logits)[0]
        )


if __name__ == "__main__":
    unittest.main()
