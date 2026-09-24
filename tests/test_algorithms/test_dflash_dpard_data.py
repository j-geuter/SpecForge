"""D-PARD requires teacher states without changing legacy DFlash datasets."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from scripts.prepare_hidden_states import parse_args, resolve_offline_capture_plan
from specforge.application import bind_run, resolve_run
from specforge.config import Config
from specforge.runtime.contracts import TrainBatch
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.training.strategies.base import DFlashTrainStrategy, StepContext


class DFlashDPARDDataTest(unittest.TestCase):
    def config(self, loss_type):
        cfg = Config(
            model={"target_model_path": "target"},
            data={"hidden_states_path": "features"},
            training={"strategy": "dflash"},
        )
        cfg.training.loss_type = loss_type
        return cfg

    def test_capture_normalization_and_collation_keep_teacher_states(self):
        cfg = self.config("dpard")
        run = resolve_run(cfg)
        run = bind_run(cfg, run.algorithm)  # CLI role rebinding is idempotent.
        provider = run.algorithm.providers.offline_for("text")
        sources = {
            "input_ids": torch.tensor([1, 2, 3]),
            "loss_mask": torch.ones(3, dtype=torch.long),
            "aux_hidden_states": torch.ones(1, 3, 8),
            "last_hidden_states": torch.arange(12).reshape(1, 3, 4),
        }
        raw = provider.capture_layout.materialize(sources)
        expected = {
            "input_ids",
            "loss_mask",
            "hidden_states",
            "target_last_hidden_states",
        }
        self.assertEqual(set(raw), expected)
        self.assertEqual(provider.capture_layout.capture_method, "dflash")
        contract = run.algorithm.spec.feature_contract("offline", "text")
        self.assertEqual(contract.required_tensors, expected)
        self.assertEqual(contract.storage.required_tensors, expected)
        normalize = provider.build_normalizer(3)
        collate = provider.build_collator()
        with tempfile.TemporaryDirectory() as directory:
            torch.save(raw, Path(directory) / "sample.ckpt")
            ref = provider.build_reader(
                directory, run_id="test", ttt_length=1, max_len=3
            ).read()[0]
            self.assertEqual(ref.strategy, "dflash")
            store = LocalFeatureStore("test")
            tensors, handle = store.get(ref)
            try:
                torch.testing.assert_close(
                    normalize(tensors)["target_last_hidden_states"],
                    sources["last_hidden_states"],
                )
            finally:
                store.release(handle, reason="test")
        batch = collate([normalize(raw), provider.build_normalizer(2)(raw)])
        self.assertEqual(batch["target_last_hidden_states"].shape, (2, 3, 4))
        torch.testing.assert_close(
            batch["target_last_hidden_states"][0], sources["last_hidden_states"][0]
        )
        self.assertEqual(batch["target_last_hidden_states"][1, 2].sum().item(), 0)

    def test_legacy_datasets_work_but_dpard_fails_lazily_on_missing_teacher(self):
        raw = {
            "input_ids": torch.tensor([1, 2, 3]),
            "loss_mask": torch.ones(3, dtype=torch.long),
            "hidden_states": torch.ones(3, 8),
        }
        with tempfile.TemporaryDirectory() as directory:
            torch.save(raw, Path(directory) / "sample.ckpt")
            for loss_type in ("dflash", "dpace", "dpard", "dpala", "dpakl"):
                with self.subTest(loss_type=loss_type):
                    provider = resolve_run(
                        self.config(loss_type)
                    ).algorithm.providers.offline_for("text")
                    with mock.patch(
                        "torch.load", side_effect=AssertionError("eager read")
                    ):
                        refs = provider.build_reader(
                            directory, run_id="test", ttt_length=1, max_len=3
                        ).read()
                    store = LocalFeatureStore("test")
                    if loss_type in {"dpard", "dpala", "dpakl"}:
                        with self.assertRaisesRegex(
                            KeyError, "target_last_hidden_states"
                        ):
                            store.get(refs[0])
                    else:
                        tensors, handle = store.get(refs[0])
                        try:
                            self.assertEqual(
                                set(provider.build_normalizer(3)(tensors)), set(raw)
                            )
                        finally:
                            store.release(handle, reason="test")

    def test_cli_loss_type_selects_four_field_capture(self):
        argv = [
            "prepare_hidden_states.py",
            "--target-model-path",
            "target",
            "--data-path",
            "data.jsonl",
            "--strategy",
            "dflash",
            "--loss-type",
            "dpard",
            "--draft-model-config",
            str(Path(__file__).resolve().parents[2] / "configs/qwen3-8b-dflash.json"),
        ]
        for objective in ("dpard", "dpala", "dpakl"):
            argv[argv.index("--loss-type") + 1] = objective
            with self.subTest(objective=objective), mock.patch("sys.argv", argv):
                args = parse_args()
                plan = resolve_offline_capture_plan(args, SimpleNamespace(num_hidden_layers=40))
                self.assertEqual(plan.capture_method, "dflash")
                self.assertIn("target_last_hidden_states", plan.layout.output_names)

    def test_dpard_teacher_reaches_non_logging_steps_with_wrapped_model(self):
        class TeacherLoss(nn.Module):
            loss_type = "dpard"

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.0))

            def forward(self, *, target_last_hidden_states=None, **kwargs):
                teacher_sum = (
                    0
                    if target_last_hidden_states is None
                    else target_last_hidden_states.sum()
                )
                return self.weight * teacher_sum, torch.tensor(0.0), {}

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = TeacherLoss()

            def forward(self, **kwargs):
                return self.module(**kwargs)

        batch = TrainBatch(
            sample_ids=["test"],
            strategy="dflash",
            tensors={
                "input_ids": torch.tensor([[1, 2, 3]]),
                "loss_mask": torch.ones(1, 3),
                "hidden_states": torch.ones(1, 3, 8),
                "target_last_hidden_states": torch.ones(1, 3, 4),
            },
        )
        for objective in ("dpard", "dpala", "dpakl"):
            for model in (TeacherLoss(), Wrapper()):
                getattr(model, "module", model).loss_type = objective
                with self.subTest(objective=objective, wrapped=isinstance(model, Wrapper)):
                    output = DFlashTrainStrategy(model).forward_loss(
                        batch, StepContext(collect_detailed_metrics=False)
                    )
                    self.assertEqual(output.loss.item(), 24.0)
                    output.loss.backward()
                    self.assertEqual(next(model.parameters()).grad.item(), 12.0)


if __name__ == "__main__":
    unittest.main()
