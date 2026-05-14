from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from lczero_training.training.utils import make_weights_mask
from proto.training_config_pb2 import OptimizerConfig

_STATES_WITH_COUNT = (
    optax.ScaleByAdamState,
    optax.ScaleByScheduleState,
)

_ABILITY_EMBEDDING_ROW_START = 112
_ABILITY_EMBEDDING_ROW_STOP = 114
_EMBEDDING_KERNEL_PATH = ("embedding", "embedding", "kernel", "value")


def update_optimizer_step(
    opt_state: optax.OptState, step: int
) -> optax.OptState:
    """Updates all step counters in the optimizer state tree."""
    step_array = jnp.array(step, dtype=jnp.int32)

    def has_count(x: object) -> bool:
        return hasattr(x, "_fields") and "count" in x._fields

    def update_count(x: optax.OptState) -> optax.OptState:
        if not has_count(x):
            return x
        assert isinstance(x, _STATES_WITH_COUNT), (
            f"Unexpected state type with 'count' field: {type(x).__name__}"
        )
        return x._replace(count=step_array)

    return jax.tree_util.tree_map(update_count, opt_state, is_leaf=has_count)


def _path_part(key: Any) -> str:
    if hasattr(key, "key"):
        return str(key.key)
    if hasattr(key, "name"):
        return str(key.name)
    if hasattr(key, "idx"):
        return str(key.idx)
    return str(key)


def _path_parts(path: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(_path_part(key) for key in path)


def _trains_embedding_kernel(config: OptimizerConfig) -> bool:
    if not config.HasField("freeze_selector"):
        return False
    return any(
        rule.match == "embedding/embedding/kernel" and not rule.include
        for rule in config.freeze_selector.rule
    )


def _mask_ability_embedding_rows() -> optax.GradientTransformation:
    """Keep only v13 ability-plane embedding rows trainable.

    The protobuf selector can only include/exclude whole leaves.  For v13
    shielded warmup we need the embedding kernel leaf to be trainable, but
    only rows 112-113 correspond to the new double-move input planes.  Masking
    gradients before clipping/Adam keeps all pretrained input and dense
    positional rows exactly frozen and prevents optimizer moments from drifting.
    """

    def init_fn(params: optax.Params) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def mask_update(path: tuple[object, ...], update: jax.Array) -> jax.Array:
        if _path_parts(path) != _EMBEDDING_KERNEL_PATH:
            return update
        if update.ndim != 2 or update.shape[0] < _ABILITY_EMBEDDING_ROW_STOP:
            raise ValueError(
                "Expected embedding/embedding/kernel to have shape "
                f"[rows, features] with at least "
                f"{_ABILITY_EMBEDDING_ROW_STOP} rows, got {update.shape}"
            )
        row_mask = jnp.zeros(update.shape[0], dtype=update.dtype)
        row_mask = row_mask.at[
            _ABILITY_EMBEDDING_ROW_START:_ABILITY_EMBEDDING_ROW_STOP
        ].set(1)
        return update * row_mask[:, None]

    def update_fn(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, optax.EmptyState]:
        del params
        return jax.tree_util.tree_map_with_path(mask_update, updates), state

    return optax.GradientTransformation(init_fn, update_fn)


def make_gradient_transformation(
    config: OptimizerConfig,
    *,
    max_grad_norm: float | None = None,
    lr_schedule: optax.Schedule,
) -> optax.GradientTransformation:
    if config.HasField("nadamw"):
        nadamw = config.nadamw
        tx = optax.nadamw(
            lr_schedule,
            b1=nadamw.beta_1,
            b2=nadamw.beta_2,
            eps=nadamw.epsilon,
            weight_decay=nadamw.weight_decay,
            mask=partial(make_weights_mask, nadamw.decay_selector),
        )
    elif config.HasField("nadam"):
        nadam = config.nadam
        tx = optax.nadam(
            lr_schedule,
            b1=nadam.beta_1,
            b2=nadam.beta_2,
            eps=nadam.epsilon,
        )
    elif config.HasField("sgd"):
        sgd = config.sgd
        tx = optax.sgd(
            lr_schedule,
            momentum=sgd.momentum if sgd.momentum else None,
            nesterov=sgd.nesterov,
        )
    else:
        raise ValueError(
            "Unsupported optimizer type: {}".format(
                config.WhichOneof("optimizer_type")
            )
        )
    if max_grad_norm is not None and max_grad_norm > 0:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
    if _trains_embedding_kernel(config):
        tx = optax.chain(_mask_ability_embedding_rows(), tx)
    if config.HasField("freeze_selector"):
        freeze_mask = partial(make_weights_mask, config.freeze_selector)

        def trainable_mask(p: nnx.State) -> nnx.State:
            return jax.tree.map(lambda x: not x, freeze_mask(p))

        tx = optax.chain(
            optax.masked(tx, trainable_mask),
            optax.masked(optax.set_to_zero(), freeze_mask),
        )
    return tx
