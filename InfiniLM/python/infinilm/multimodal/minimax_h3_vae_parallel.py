"""Opt-in spatial tile scheduling; original temporal decode and stitching stay in use."""
from __future__ import annotations

import copy
import json
import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Tile:
    index: int
    row: int
    column: int
    top: int
    bottom: int
    left: int
    right: int


def tile_plan(vae, latent_height, latent_width):
    ratio = vae.spatial_compression_ratio
    ys, hs, yo = vae._split_tiles(latent_height * ratio, vae.tile_sample_min_height,
                                  vae.tile_sample_min_overlap_height)
    xs, ws, xo = vae._split_tiles(latent_width * ratio, vae.tile_sample_min_width,
                                  vae.tile_sample_min_overlap_width)
    values = [*ys, *hs, *xs, *ws, *yo, *xo]
    if ratio <= 0 or any(v % ratio for v in values):
        raise ValueError("VAE tile boundaries must remain latent aligned")
    tiles = [Tile(i * len(xs) + j, i, j, y // ratio, (y + h) // ratio,
                  x // ratio, (x + w) // ratio)
             for i, (y, h) in enumerate(zip(ys, hs))
             for j, (x, w) in enumerate(zip(xs, ws))]
    return tiles, len(ys), len(xs), yo, xo


def distribute(tiles, workers):
    if workers <= 0:
        raise ValueError("At least one tile worker is required")
    return [tiles[i::workers] for i in range(workers)]


def assemble(results, rows, columns):
    flat = {}
    for group in results:
        for index, value in group:
            if index in flat:
                raise ValueError("Duplicate tile result")
            flat[index] = value
    if set(flat) != set(range(rows * columns)):
        raise ValueError("Missing or unexpected tile result")
    return [[flat[i * columns + j] for j in range(columns)] for i in range(rows)]


def clone_to(module, device):
    # Copy tensors directly to the destination, preserving tied tensors and dtype.
    memo = {}
    for tensor in [*module.parameters(), *module.buffers()]:
        if id(tensor) not in memo:
            value = tensor.detach().to(device=device, copy=True)
            memo[id(tensor)] = (torch.nn.Parameter(value, requires_grad=False)
                                if isinstance(tensor, torch.nn.Parameter) else value)
    return copy.deepcopy(module, memo).requires_grad_(False).eval()


def decode_tiles(post_quant_conv, decoder, latent, tiles):
    # Batch stays exactly as upstream. Do not combine tiles along the batch axis.
    return [(t.index, decoder(post_quant_conv(latent[..., t.top:t.bottom, t.left:t.right])))
            for t in tiles]


class ParallelTileDecoder:
    """One fixed thread/stream per device; two-phase compute then gather."""

    def __init__(self, vae, devices, reserve_mib=2048):
        if getattr(vae, "_phase8_parallel_decode", None) is not None:
            raise ValueError("Parallel VAE decoding is already installed")
        main = next(vae.decoder.parameters()).device
        if main.type != "cuda" or main.index not in devices:
            raise ValueError("Requested devices must contain the VAE execution device")
        if vae.training or any(m.training for m in (vae.decoder, vae.post_quant_conv)):
            raise ValueError("Parallel VAE decode only supports eval inference")
        if len(devices) < 2 or len(set(devices)) != len(devices) or reserve_mib < 0:
            raise ValueError("Use at least two distinct devices and a nonnegative reserve")
        for root in (vae.decoder, vae.post_quant_conv):
            if any(getattr(m, "_hf_hook", None) is not None for m in root.modules()):
                raise ValueError("Accelerate offload hooks on decoder replicas are unsupported")
        self.vae, self.main = vae, main
        self.devices = [main.index] + [d for d in devices if d != main.index]
        self.reserve = reserve_mib * 1024**2
        self.pools, self.replicas = [], []
        self.lock = threading.Lock()
        self.closed, self.failed = False, False
        self.original = vae._decode_clip
        self.had_instance = "_decode_clip" in vars(vae)
        self.instance_value = vars(vae).get("_decode_clip")
        self.wrapper = types.MethodType(lambda owner, z: self.decode_clip(z), vae)
        vae._decode_clip = self.wrapper
        vae._phase8_parallel_decode = self

    def _make_replica(self, device):
        with torch.cuda.device(device), torch.inference_mode():
            free, total = torch.cuda.mem_get_info(device)
            weight_bytes = sum(t.numel() * t.element_size()
                               for root in (self.vae.decoder, self.vae.post_quant_conv)
                               for t in [*root.parameters(), *root.buffers()])
            required = self.reserve + (0 if device == self.main.index else weight_bytes)
            if free < required:
                raise RuntimeError(f"VAE device {device}: free={free}, required={required}")
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                decoder = self.vae.decoder if device == self.main.index else clone_to(
                    self.vae.decoder, torch.device("cuda", device))
                post = self.vae.post_quant_conv if device == self.main.index else clone_to(
                    self.vae.post_quant_conv, torch.device("cuda", device))
                # Create each thread/device's MIOpen handle serially, before peers compute.
                x = torch.zeros(1, 1, 1, 1, 1, device=device)
                torch.nn.functional.conv3d(x, torch.zeros_like(x))
            stream.synchronize()
            print("PHASE8_VAE_REPLICA_READY", json.dumps(dict(
                device=device, free_before=free, total=total, weights=weight_bytes)), flush=True)
            return device, stream, post, decoder

    def _prepare(self):
        if self.replicas:
            return
        # Serialized module initialization avoids lazy HIP/MIOpen loads during peer kernels.
        for device in self.devices:
            pool = ThreadPoolExecutor(1, thread_name_prefix=f"h3-phase8-vae-{device}")
            self.pools.append(pool)
            self.replicas.append(pool.submit(self._make_replica, device).result())

    @staticmethod
    def _run(replica, z, tiles, autocast):
        device, stream, post, decoder = replica
        enabled, dtype, cache = autocast
        with torch.cuda.device(device), torch.cuda.stream(stream), torch.inference_mode():
            local = z if z.device.index == device else z.to(device=device)
            with torch.autocast("cuda", dtype=dtype, enabled=enabled, cache_enabled=cache):
                result = decode_tiles(post, decoder, local, tiles)
            stream.synchronize()
            return result

    def decode_clip(self, z):
        with self.lock:
            if self.closed or self.failed:
                raise RuntimeError("Parallel VAE decoder is closed or previously failed")
            if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
                raise RuntimeError("Parallel VAE requires no_grad/inference_mode")
            if not self.vae.use_tiling:
                return self.original(z)
            tiles, rows, columns, yo, xo = tile_plan(self.vae, z.shape[-2], z.shape[-1])
            if len(tiles) < 2:
                return self.original(z)
            if z.device != self.main:
                raise ValueError("Latents must be on the original VAE execution device")
            try:
                # Publish all producer work before any thread accesses z or shared weights.
                torch.cuda.current_stream(self.main).synchronize()
                self._prepare()
                autocast = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
                            torch.is_autocast_cache_enabled())
                groups = distribute(tiles, len(self.replicas))
                pending = [pool.submit(self._run, rep, z, part, autocast)
                           for pool, rep, part in zip(self.pools, self.replicas, groups)]
                results, failure = [], None
                for future in pending:
                    try:
                        results.append(future.result())
                    except BaseException as exc:
                        failure = failure or exc
                if failure is not None:
                    raise failure
                # All producers have synchronized. Blocking copies cannot form a cross-stream cycle.
                with torch.cuda.device(self.main):
                    gathered = [[(i, t.to(device=self.main)) for i, t in group] for group in results]
                    for group in gathered:
                        for _, tensor in group:
                            tensor.record_stream(torch.cuda.current_stream(self.main))
                    grid = assemble(gathered, rows, columns)
                    return self.vae._stitch_tiles(grid, yo, xo)
            except BaseException:
                self.failed = True
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            replaced = self.vae._decode_clip is not self.wrapper
            if not replaced:
                if self.had_instance:
                    self.vae._decode_clip = self.instance_value
                else:
                    delattr(self.vae, "_decode_clip")
            if getattr(self.vae, "_phase8_parallel_decode", None) is self:
                delattr(self.vae, "_phase8_parallel_decode")
            try:
                for pool in self.pools:
                    pool.shutdown(wait=True)
            finally:
                self.pools.clear()
                self.replicas.clear()
                self.closed = True
            if replaced:
                raise RuntimeError("Another owner replaced _decode_clip; preserved it and closed workers")


def install_parallel_vae_decode(vae, devices):
    count = int(os.environ.get("H3_EXPERIMENT_VAE_DEVICES", "0"))
    if count in (0, 1):
        return None
    if count not in (2, 4, 8) or count > len(devices):
        raise ValueError("H3_EXPERIMENT_VAE_DEVICES must be 0/1/2/4/8 within selected GPUs")
    reserve = int(os.environ.get("H3_EXPERIMENT_VAE_RESERVE_MIB", "2048"))
    return ParallelTileDecoder(vae, list(devices[:count]), reserve)
