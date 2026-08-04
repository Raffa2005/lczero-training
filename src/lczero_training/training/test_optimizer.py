import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from lczero_training.training.optimizer import (
    make_gradient_transformation,
    _mask_embedding_kernel_rows,
    _trains_embedding_kernel,
)
from proto import training_config_pb2 as pb


def _updates() -> dict:
    return {
        "embedding": {
            "embedding": {
                "kernel": {"value": jnp.ones((242, 4), dtype=jnp.float32)},
            },
        },
        "other": {"value": jnp.ones((3,), dtype=jnp.float32)},
    }


class _InputEmbedding(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.embedding = nnx.Linear(
            in_features=242,
            out_features=4,
            rngs=rngs,
        )


class _TinyModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.embedding = _InputEmbedding(rngs=rngs)


def _optimizer_config(*, all_rows: bool) -> pb.OptimizerConfig:
    config = pb.OptimizerConfig(
        nadamw=pb.NadamwOptimizerConfig(
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        ),
    )
    config.freeze_selector.rule.add(match="embedding/**", include=False)
    config.freeze_selector.otherwise_include = True
    if all_rows:
        config.embedding_kernel_row_mode = pb.OptimizerConfig.ALL_ROWS
    return config


def test_ability_row_mode_masks_standard_and_dense_rows() -> None:
    tx = _mask_embedding_kernel_rows(train_all_rows=False)
    updates = _updates()
    masked, _ = tx.update(updates, tx.init(updates))

    kernel = np.asarray(masked["embedding"]["embedding"]["kernel"]["value"])
    assert np.count_nonzero(kernel) == 8
    np.testing.assert_array_equal(kernel[112:114], np.ones((2, 4)))
    np.testing.assert_array_equal(masked["other"]["value"], np.ones((3,)))


def test_all_row_mode_is_identity() -> None:
    tx = _mask_embedding_kernel_rows(train_all_rows=True)
    updates = _updates()
    masked, _ = tx.update(updates, tx.init(updates))

    np.testing.assert_array_equal(
        masked["embedding"]["embedding"]["kernel"]["value"],
        updates["embedding"]["embedding"]["kernel"]["value"],
    )


def test_embedding_kernel_detection_honors_wildcard_selector() -> None:
    config = pb.OptimizerConfig()
    config.freeze_selector.rule.add(match="embedding/**", include=False)
    config.freeze_selector.otherwise_include = True

    assert _trains_embedding_kernel(config)


def test_embedding_kernel_detection_honors_first_matching_rule() -> None:
    config = pb.OptimizerConfig()
    config.freeze_selector.rule.add(
        match="embedding/embedding/kernel", include=True
    )
    config.freeze_selector.rule.add(match="embedding/**", include=False)
    config.freeze_selector.otherwise_include = True

    assert not _trains_embedding_kernel(config)


def test_missing_freeze_selector_keeps_general_optimizer_behavior() -> None:
    assert not _trains_embedding_kernel(pb.OptimizerConfig())


def test_row_modes_keep_optimizer_state_structure_and_change_only_updates() -> (
    None
):
    params = nnx.state(_TinyModel(rngs=nnx.Rngs(0)))
    grads = jax.tree.map(jnp.ones_like, params)
    transformations = [
        make_gradient_transformation(
            _optimizer_config(all_rows=all_rows),
            lr_schedule=lambda _step: jnp.asarray(1e-3),
        )
        for all_rows in (False, True)
    ]
    states = [tx.init(params) for tx in transformations]

    assert jax.tree.structure(states[0]) == jax.tree.structure(states[1])

    protected_updates, _ = transformations[0].update(grads, states[0], params)
    full_updates, _ = transformations[1].update(grads, states[1], params)
    protected_kernel = np.asarray(
        protected_updates["embedding"]["embedding"]["kernel"]
    )
    full_kernel = np.asarray(full_updates["embedding"]["embedding"]["kernel"])

    assert np.count_nonzero(protected_kernel) == 8
    assert np.count_nonzero(full_kernel) == 242 * 4
