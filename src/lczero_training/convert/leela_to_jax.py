import dataclasses
import gzip
import logging
import math
from typing import Any, Optional, cast

import jax.numpy as jnp
from flax import nnx, serialization

from lczero_training.model.model import LczeroModel
from proto import hlo_pb2, net_pb2

from .jax_to_leela import LeelaExportOptions, jax_to_leela
from .leela_pytree_visitor import LeelaPytreeWeightsVisitor
from .leela_to_modelconfig import leela_to_modelconfig

logger = logging.getLogger(__name__)


_EMBEDDING_PLANE_TO_SCALE = 109
_EMBEDDING_SCALE = 99.0


@dataclasses.dataclass
class LeelaImportOptions:
    weights_dtype: hlo_pb2.XlaShapeProto.Type
    compute_dtype: hlo_pb2.XlaShapeProto.Type


def fix_older_weights_file(file: net_pb2.Net) -> None:
    nf = net_pb2.NetworkFormat
    has_network_format = file.format.HasField("network_format")
    network_format = (
        file.format.network_format.network if has_network_format else None
    )

    net = file.format.network_format

    if not has_network_format:
        # Older protobufs don't have format definition.
        net.input = nf.INPUT_CLASSICAL_112_PLANE
        net.output = nf.OUTPUT_CLASSICAL
        net.network = nf.NETWORK_CLASSICAL_WITH_HEADFORMAT
        net.value = nf.VALUE_CLASSICAL
        net.policy = nf.POLICY_CLASSICAL
    elif network_format == nf.NETWORK_CLASSICAL:
        # Populate policyFormat and valueFormat fields in old protobufs
        # without these fields.
        net.network = nf.NETWORK_CLASSICAL_WITH_HEADFORMAT
        net.value = nf.VALUE_CLASSICAL
        net.policy = nf.POLICY_CLASSICAL
    elif network_format == nf.NETWORK_SE:
        net.network = nf.NETWORK_SE_WITH_HEADFORMAT
        net.value = nf.VALUE_CLASSICAL
        net.policy = nf.POLICY_CLASSICAL
    elif (
        network_format == nf.NETWORK_SE_WITH_HEADFORMAT
        and len(file.weights.encoder) > 0
    ):
        # Attention body network made with old protobuf.
        net.network = nf.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
        if file.weights.HasField("smolgen_w"):
            # Need to override activation defaults for smolgen.
            net.ffn_activation = nf.ACTIVATION_RELU_2
            net.smolgen_activation = nf.ACTIVATION_SWISH
    elif network_format == nf.NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT:
        net.network = nf.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT

    if (
        file.format.network_format.network
        == nf.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
    ):
        weights = file.weights
        if weights.HasField("policy_heads") and weights.HasField("value_heads"):
            logger.info(
                "Weights file has multihead format, updating format flag"
            )
            net.network = nf.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
            net.input_embedding = nf.INPUT_EMBEDDING_PE_DENSE
        if not file.format.network_format.HasField("input_embedding"):
            net.input_embedding = nf.INPUT_EMBEDDING_PE_MAP


class LeelaToJax(LeelaPytreeWeightsVisitor):
    def embedding_block(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        # Replicates super().embedding_block(...) with the embedding-kernel
        # matmul replaced by _extend_embedding_kernel: standard lc0 oracles
        # have 112 input planes, doublemove trainee has 114 (planes 112-113
        # carry "our/their ability available"). Splice 2 zero rows between
        # the standard-plane rows and the positional-encoding rows.
        self.matmul(
            nnx_dict["preprocess"],
            weights.ip_emb_preproc_w,
            weights.ip_emb_preproc_b,
        )
        self._extend_embedding_kernel(
            nnx_dict["embedding"]["kernel"], weights.ip_emb_w
        )
        self.tensor(nnx_dict["embedding"]["bias"], weights.ip_emb_b)
        self.layernorm(
            nnx_dict["norm"],
            weights.ip_emb_ln_gammas,
            weights.ip_emb_ln_betas,
        )
        self.tensor(
            nnx_dict["ma_gating"]["mult_gate"]["gate"], weights.ip_mult_gate
        )
        self.tensor(
            nnx_dict["ma_gating"]["add_gate"]["gate"], weights.ip_add_gate
        )
        self.ffn(nnx_dict["ffn"], weights.ip_emb_ffn)
        self.layernorm(
            nnx_dict["out_norm"],
            weights.ip_emb_ffn_ln_gammas,
            weights.ip_emb_ffn_ln_betas,
        )

        # Plane 109 (rule50) lives in [0..111] so the splice doesn't move
        # it. Same scaling as before.
        embedding_kernel = cast(nnx.Param, nnx_dict["embedding"]["kernel"])
        values = embedding_kernel.value
        scaled_values = values.at[_EMBEDDING_PLANE_TO_SCALE].set(
            values[_EMBEDDING_PLANE_TO_SCALE] * _EMBEDDING_SCALE
        )
        embedding_kernel.value = scaled_values

    def _extend_embedding_kernel(
        self,
        kernel_param: Any,
        oracle_layer: net_pb2.Weights.Layer,
    ) -> None:
        """Load oracle ip_emb_w with 2 zero-init rows for ability planes.

        Oracle kernel shape: (112 + dense_size, embedding_size).
        Trainee kernel shape: (114 + dense_size, embedding_size).

        Layout after splice (rows = input features):
            rows 0..111      ← oracle rows 0..111   (standard planes)
            rows 112..113    ← zeros                (ability planes — new)
            rows 114..114+ds ← oracle rows 112..    (dense positional rows)

        Bias is unaffected (its shape is (embedding_size,)).
        """
        trainee_in, embedding_size = kernel_param.shape
        oracle_total = len(oracle_layer.params) // 2
        assert oracle_total != 0, "Oracle ip_emb_w is empty"
        assert oracle_total % embedding_size == 0, (
            f"Oracle ip_emb_w has {oracle_total} values, not divisible by "
            f"embedding_size={embedding_size}"
        )
        oracle_in = oracle_total // embedding_size
        assert oracle_in >= 112, (
            f"Oracle ip_emb_w has {oracle_in} input rows; expected >=112"
        )
        dense_size = oracle_in - 112
        assert trainee_in == 114 + dense_size, (
            f"Trainee embedding kernel input dim is {trainee_in}; "
            f"expected 114 + dense_size = {114 + dense_size} "
            f"(dense_size derived from oracle = {dense_size})"
        )

        # Decode oracle weights identically to the base tensor():
        # uint16 LINEAR16 → float32 → reshape (out, in).T → (in, out).
        raw = jnp.frombuffer(oracle_layer.params, dtype=jnp.uint16).astype(
            jnp.float32
        )
        alpha = raw / 65535.0
        decoded = (
            alpha * oracle_layer.max_val
            + (1.0 - alpha) * oracle_layer.min_val
        )
        oracle_kernel = decoded.reshape(embedding_size, oracle_in).transpose()

        zeros = jnp.zeros((2, embedding_size), dtype=oracle_kernel.dtype)
        new_kernel = jnp.concatenate(
            [
                oracle_kernel[:112],   # standard planes 0..111
                zeros,                 # ability planes 112, 113 (zero-init)
                oracle_kernel[112:],   # dense positional rows 114..
            ],
            axis=0,
        )
        assert new_kernel.shape == kernel_param.shape

        kernel_param.value = new_kernel.astype(kernel_param.dtype)

    def tensor(
        self,
        param: nnx.Param,
        leela: net_pb2.Weights.Layer,
    ) -> None:
        assert len(leela.params) // 2 == math.prod(param.shape)
        assert len(leela.params) != 0

        values = jnp.frombuffer(leela.params, dtype=jnp.uint16)
        values = values.astype(jnp.float32)
        alpha = values / 65535.0
        values = alpha * leela.max_val + (1.0 - alpha) * leela.min_val
        values = values.astype(param.dtype)
        values = values.reshape(param.shape[::-1]).transpose()
        param.value = values


def leela_to_jax(
    leela_net: net_pb2.Net, import_options: LeelaImportOptions
) -> nnx.State:
    config = leela_to_modelconfig(
        leela_net,
        import_options.weights_dtype,
        import_options.compute_dtype,
    )

    model = LczeroModel(config=config, rngs=nnx.Rngs(params=42))
    state = nnx.state(model)
    visitor = LeelaToJax(state, leela_net)
    visitor.run()

    return state


def leela_to_jax_files(
    input_path: str,
    weights_dtype: str,
    compute_dtype: str,
    output_modelconfig: Optional[str],
    output_serialized_jax: Optional[str],
    output_leela_verification: Optional[str],
    print_modelconfig: bool = False,
) -> None:
    lc0_weights = net_pb2.Net()
    with gzip.open(input_path, "rb") as f:
        contents = f.read()
        assert isinstance(contents, bytes)
        lc0_weights.ParseFromString(contents)

    fix_older_weights_file(lc0_weights)

    import_options = LeelaImportOptions(
        weights_dtype=getattr(hlo_pb2.XlaShapeProto, weights_dtype),
        compute_dtype=getattr(hlo_pb2.XlaShapeProto, compute_dtype),
    )

    config = leela_to_modelconfig(
        lc0_weights,
        import_options.weights_dtype,
        import_options.compute_dtype,
    )

    if print_modelconfig:
        print(config)

    if output_modelconfig:
        with open(output_modelconfig, "w") as f:
            f.write(str(config))

    if output_serialized_jax is None and output_leela_verification is None:
        return

    state = leela_to_jax(lc0_weights, import_options)

    if output_serialized_jax:
        with open(output_serialized_jax, "wb") as f:
            f.write(serialization.to_bytes(state))

    if output_leela_verification:
        min_version = (
            f"v{lc0_weights.min_version.major}."
            f"{lc0_weights.min_version.minor}."
            f"{lc0_weights.min_version.patch}"
        )
        license_str = (
            lc0_weights.license if lc0_weights.HasField("license") else None
        )
        export_options = LeelaExportOptions(
            min_version=min_version,
            num_heads=lc0_weights.weights.headcount,
            license=license_str,
            training_steps=lc0_weights.training_params.training_steps,
        )
        verification_net = jax_to_leela(state, export_options)
        with gzip.open(output_leela_verification, "wb") as f:
            f.write(verification_net.SerializeToString())
