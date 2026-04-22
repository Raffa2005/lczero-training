import argparse
import datetime
import gzip
import logging
import os
import sys

import jax
import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format

from lczero_training.commands import configure_root_logging
from lczero_training.convert.jax_to_leela import (
    LeelaExportOptions,
    jax_to_leela,
)
from lczero_training.dataloader import make_dataloader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.lr_schedule import make_lr_schedule
from lczero_training.training.optimizer import make_gradient_transformation
from lczero_training.training.state import TrainingState
from lczero_training.training.training import Training, from_dataloader
from proto.root_config_pb2 import RootConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start a training run.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    return parser


def _enable_jax_persistent_cache() -> None:
    # Persist XLA HLO compilation across train runs. Each `lc0-train` invocation
    # starts a fresh Python process, so without this the train_step is
    # JIT-compiled from scratch every iteration (~8s on T4). The cache lives at
    # $LC0_JAX_CACHE_DIR or ~/.cache/lczero-jax-jit by default.
    cache_dir = os.environ.get(
        "LC0_JAX_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "lczero-jax-jit"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    logging.info("JAX persistent compilation cache: %s", cache_dir)


def train(config_filename: str) -> None:
    _enable_jax_persistent_cache()
    config = RootConfig()
    logging.info("Reading configuration from proto file")
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if config.training.checkpoint.path is None:
        logging.error("Checkpoint path must be set in the configuration.")
        sys.exit(1)

    checkpoint_mgr = ocp.CheckpointManager(
        config.training.checkpoint.path,
        options=ocp.CheckpointManagerOptions(
            create=True,
        ),
    )

    logging.info("Creating state from configuration")
    empty_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )
    latest_step = checkpoint_mgr.latest_step()
    if latest_step is not None:
        logging.info("Restoring checkpoint at step %s", latest_step)
        training_state = checkpoint_mgr.restore(
            latest_step, args=ocp.args.PyTreeRestore(empty_state)
        )
    else:
        logging.info("No checkpoint found, starting from scratch")
        # Deep-copy to ensure model_state and swa_state are separate buffers,
        # since JAX donation fails if the same buffer appears twice.
        training_state = jax.tree.map(lambda x: x.copy() if hasattr(x, 'copy') else x, empty_state)
    logging.info("Restored checkpoint")

    model, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )

    assert isinstance(training_state, TrainingState)

    jit_state = training_state.jit_state
    lr_sched = make_lr_schedule(config.training.lr_schedule)
    optimizer_tx = make_gradient_transformation(
        config.training.optimizer,
        max_grad_norm=getattr(config.training, "max_grad_norm", 0.0),
        lr_schedule=lr_sched,
    )
    training = Training(
        optimizer_tx=optimizer_tx,
        graphdef=model,
        loss_fn=LczeroLoss(config=config.training.losses),
        swa_config=(
            config.training.swa if config.training.HasField("swa") else None
        ),
    )
    new_state = training.run(
        jit_state,
        from_dataloader(make_dataloader(config.data_loader)),
        config.training.schedule.steps_per_network,
    )

    training_state = training_state.replace(jit_state=new_state)
    logging.info("Saving checkpoint at step %d", new_state.step)
    checkpoint_mgr.save(
        step=new_state.step,
        args=ocp.args.PyTreeSave(item=training_state),
    )
    checkpoint_mgr.wait_until_finished()
    logging.info("Checkpoint saved")

    if config.export.destination_filename:
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        logging.info("Exporting network")

        options = LeelaExportOptions(
            min_version="0.28",
            num_heads=training_state.num_heads,
            license=None,
            training_steps=new_state.step,
        )
        export_state = (
            new_state.swa_state
            if config.export.export_swa_model
            else new_state.model_state
        )
        assert isinstance(export_state, nnx.State)
        net = jax_to_leela(jax_weights=export_state, export_options=options)
        network_bytes = gzip.compress(net.SerializeToString())

        for destination_template in config.export.destination_filename:
            destination = destination_template.format(
                datetime=date_str, step=new_state.step
            )
            logging.info(f"Writing network to {destination}")
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, "wb") as f:
                f.write(network_bytes)
            logging.info(f"Finished writing network to {destination}")


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    train(config_filename=args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
