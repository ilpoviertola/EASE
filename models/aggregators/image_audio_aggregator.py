from .base_aggregator import BaseAggregator, masked_max, masked_mean


class ImageThenAudioAggregator(BaseAggregator):
    def __init__(
        self,
        image_agg_type,
        audio_agg_type,
        nonneg_sim,
        mask_silence,
        num_heads,
        head_agg,
        use_cls,
    ):
        super().__init__(nonneg_sim, mask_silence, num_heads, head_agg, use_cls)
        if image_agg_type == "max":
            self.image_agg = lambda x, dim: x.max(dim=dim, keepdim=True).values
        elif image_agg_type == "avg":
            self.image_agg = lambda x, dim: x.mean(dim=dim, keepdim=True)
        else:
            raise ValueError(f"Unknown image_agg_type {image_agg_type}")

        if audio_agg_type == "max":
            self.time_agg = masked_max
        elif audio_agg_type == "avg":
            self.time_agg = masked_mean
        else:
            raise ValueError(f"Unknown audio_agg_type {audio_agg_type}")

        self.freq_agg = lambda x, dim: x.mean(dim=dim, keepdim=True)

    def _agg_sim(self, sim, mask):
        sim_shape = sim.shape
        new_mask_shape = [1] * len(sim_shape)
        new_mask_shape[0] = sim_shape[0]
        new_mask_shape[-1] = sim_shape[-1]
        mask = mask.reshape(new_mask_shape)
        sim = self.image_agg(sim, -3)
        sim = self.image_agg(sim, -4)
        sim = self.freq_agg(sim, -2)
        sim = self.time_agg(sim, mask, -1)
        return sim.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)
