# Double Strike Chess Training

JAX/Flax training pipeline adapted from
[Leela Chess Zero](https://github.com/LeelaChessZero/lczero-training).
It supports the ability input planes, split policy heads, gate training, and
fine-tuning of pretrained transformer networks.

## Build

Use Linux, Python 3.13+, a C++20 compiler, Meson, Ninja, `uv`, and `just`.
The Python dependency set targets CUDA 12. Install a compatible GPU driver.

```sh
git submodule update --init --recursive
uv sync
just build-proto
uv run meson setup build/release --buildtype=release --native-file=native.ini
uv run meson compile -C build/release -j 4
just mksymlink
```

Create the virtual environment with `uv sync` before configuring Meson so that
the extension is built against the correct Python interpreter.

## Commands

```sh
uv run lc0-init --help
uv run lc0-leela2jax --help
uv run lc0-train --help
uv run lc0-jax2leela --help
```

Initialize or import a checkpoint, configure the data loader with compatible
self-play chunks, then run `lc0-train --config /path/to/config.textproto`.
See `docs/` and `proto/training_config.proto` for the configuration schema and
loader architecture. Training configuration controls checkpoint/export paths,
replay selection, parameter freezing, losses, and optimizer settings.

## Tests

```sh
uv run meson test -C build/release
uv run pytest
```

## Attribution

The upstream training implementation is by the Leela Chess Zero contributors.
Rafael Sabirov's original Double Strike Chess modifications are offered under
`LICENSE-CONTRIBUTIONS`. That grant covers those contributions only and does
not relicense upstream files. The nested lc0 source retains its GPL license.
