"""Let OffloadingConnector sync_load run with per-group block sizes (MLA 1:2:4 pages).

sync_load locates each group's load boundary positionally from token counts
(scheduler.update_state_after_alloc), which is exact whenever the GPU-computed
prefix is a whole number of blocks in every group. Instead of refusing
non-uniform block sizes at startup, skip the offload lookup for the rare
request whose GPU hit is not aligned that way (e.g. partial hash hits).
"""


def apply() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as s

    if getattr(s, "_qwen_mla_sync_multiblock", False):
        return
    s._qwen_mla_sync_multiblock = True

    orig_from_spec = s.SchedulerOffloadConfig.from_spec.__func__

    def from_spec(cls, spec, vllm_config, kv_cache_config):
        sync_load = spec.sync_load
        spec.sync_load = False  # bypass the uniform-block-size guard only
        try:
            cfg = orig_from_spec(cls, spec, vllm_config, kv_cache_config)
        finally:
            spec.sync_load = sync_load
        return cfg._replace(sync_load=sync_load)

    s.SchedulerOffloadConfig.from_spec = classmethod(from_spec)

    orig_lookup = s.OffloadingConnectorScheduler._lookup

    def _lookup(self, req_status):
        n = req_status.num_locally_computed_tokens
        if self.config.sync_load and any(
            n % g.tokens_per_block for g in self.config.kv_group_configs
        ):
            return 0
        return orig_lookup(self, req_status)

    s.OffloadingConnectorScheduler._lookup = _lookup
