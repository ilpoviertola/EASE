import math
from abc import abstractmethod

import torch
from tqdm import tqdm

AUDIO_CLS = "audio_cls"
AUDIO_FEATS = "audio_feats"
AUDIO_MASK = "audio_mask"
IMAGE_CLS = "image_cls"
IMAGE_FEATS = "image_feats"


@torch.jit.script
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int):
    mask = mask.to(x)
    return (x * mask).sum(dim, keepdim=True) / mask.sum(dim, keepdim=True).clamp_min(
        0.001
    )


@torch.jit.script
def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int):
    mask = mask.to(torch.bool)
    eps = 1e7
    # eps = torch.finfo(x.dtype).max
    return (x - (~mask) * eps).max(dim, keepdim=True).values


def masked_lse(x: torch.Tensor, mask: torch.Tensor, dim: int, temp):
    x = x.to(torch.float32)
    mask = mask.to(torch.float32)
    x_masked = x - (1 - mask) * torch.finfo(x.dtype).max
    return (
        torch.logsumexp(x_masked * temp, dim, keepdim=True)
        - torch.log(mask.sum(dim, keepdim=True))
    ) / temp


class BaseAggregator(torch.nn.Module):

    def __init__(self, nonneg_sim, mask_silence, num_heads, head_agg, use_cls):
        super().__init__()

        self.nonneg_sim = nonneg_sim
        self.mask_silence = mask_silence
        self.num_heads = num_heads
        self.head_agg = head_agg
        self.use_cls = use_cls

    @abstractmethod
    def _agg_sim(self, sim, mask):
        pass

    def prepare_sims(self, sim, mask, agg_sim, agg_heads):
        sim_size = sim.shape
        assert len(mask.shape) == 2
        assert len(sim_size) in {
            6,
            7,
        }, f"sim has wrong number of dimensions: {sim.shape}"
        pairwise = len(sim_size) == 6

        if self.mask_silence:
            mask = mask
        else:
            mask = torch.ones_like(mask)

        if self.nonneg_sim:
            sim = sim.clamp_min(0)

        if pairwise:
            head_dim = 1
        else:
            head_dim = 2

        if self.head_agg == "max_elementwise" and agg_heads:
            sim = sim.max(head_dim, keepdim=True).values

        if agg_sim:
            sim = self._agg_sim(sim, mask)

        if agg_heads:
            if self.head_agg == "sum" or self.head_agg == "max_elementwise":
                sim = sim.sum(head_dim)
            elif self.head_agg == "max":
                sim = sim.max(head_dim).values
            else:
                raise ValueError(f"Unknown head_agg: {self.head_agg}")

        return sim

    def _get_full_sims(self, preds, raw, agg_sim, agg_heads):
        if agg_sim or agg_heads or raw:
            assert (
                agg_sim or agg_heads
            ) != raw, "Cannot have raw on at the same time as agg_sim or agg_heads"

        audio_feats = preds[AUDIO_FEATS]
        audio_mask = preds[AUDIO_MASK]
        image_feats = preds[IMAGE_FEATS]

        b1, c2, f, t1 = audio_feats.shape
        b2, t2 = audio_mask.shape
        d, c1, h, w = image_feats.shape
        assert b1 == b2 and c1 == c2 and t1 == t2
        assert c1 % self.num_heads == 0
        new_c = c1 // self.num_heads
        audio_feats = audio_feats.reshape(b1, self.num_heads, new_c, f, t1)
        image_feats = image_feats.reshape(d, self.num_heads, new_c, h, w)
        raw_sims = torch.einsum(
            "akcft,vkchw->avkhwft",
            audio_feats.to(torch.float32),
            image_feats.to(torch.float32),
        )

        if self.use_cls:
            audio_cls = preds[AUDIO_CLS].reshape(b1, self.num_heads, new_c)
            image_cls = preds[IMAGE_CLS].reshape(d, self.num_heads, new_c)
            cls_sims = torch.einsum(
                "akc,vkc->avk", audio_cls.to(torch.float32), image_cls.to(torch.float32)
            )
            raw_sims += cls_sims.reshape(b1, d, self.num_heads, 1, 1, 1, 1)

        if raw:
            return raw_sims
        else:
            return self.prepare_sims(raw_sims, audio_mask, agg_sim, agg_heads)

    def get_pairwise_sims(self, preds, raw, agg_sim, agg_heads) -> torch.Tensor:
        if agg_sim or agg_heads or raw:
            assert (
                agg_sim or agg_heads
            ) != raw, "Cannot have raw on at the same time as agg_sim or agg_heads"

        audio_feats = preds[AUDIO_FEATS]
        audio_mask = preds[AUDIO_MASK]
        image_feats = preds[IMAGE_FEATS]

        a1, c1, f, t1 = audio_feats.shape
        a2, t2 = audio_mask.shape

        assert c1 % self.num_heads == 0
        new_c = c1 // self.num_heads
        audio_feats = audio_feats.reshape(a1, self.num_heads, new_c, f, t1)

        if len(image_feats.shape) == 5:
            print("Using similarity for video, should only be called during plotting")
            v, vt, c2, h, w = image_feats.shape
            image_feats = image_feats.reshape(v, vt, self.num_heads, new_c, h, w)
            raw_sims = torch.einsum(
                "bkcft,bskchw,bt->bskhwft",
                audio_feats.to(torch.float32),
                image_feats.to(torch.float32),
                audio_mask.to(torch.float32),
            )

            if self.use_cls:
                audio_cls = preds[AUDIO_CLS].reshape(v, self.num_heads, new_c)
                image_cls = preds[IMAGE_CLS].reshape(v, vt, self.num_heads, new_c)
                cls_sims = torch.einsum(
                    "bkc,bskc->bsk",
                    audio_cls.to(torch.float32),
                    image_cls.to(torch.float32),
                )
                raw_sims += cls_sims.reshape(v, vt, self.num_heads, 1, 1, 1, 1)

        elif len(image_feats.shape) == 4:
            v, c2, h, w = image_feats.shape
            image_feats = image_feats.reshape(v, self.num_heads, new_c, h, w)
            raw_sims = torch.einsum(
                "bkcft,bkchw,bt->bkhwft",
                audio_feats.to(torch.float32),
                image_feats.to(torch.float32),
                audio_mask.to(torch.float32),
            )

            if self.use_cls:
                audio_cls = preds[AUDIO_CLS].reshape(v, self.num_heads, new_c)
                image_cls = preds[IMAGE_CLS].reshape(v, self.num_heads, new_c)
                cls_sims = torch.einsum(
                    "bkc,bkc->bk",
                    audio_cls.to(torch.float32),
                    image_cls.to(torch.float32),
                )
                raw_sims += cls_sims.reshape(v, self.num_heads, 1, 1, 1, 1)
        else:
            raise ValueError(f"Improper image shape: {image_feats.shape}")

        assert a1 == a2 and c2 == c2 and t1 == t2

        if raw:
            return raw_sims
        else:
            return self.prepare_sims(raw_sims, audio_mask, agg_sim, agg_heads)

    def forward(self, preds, agg_heads):
        return self._get_full_sims(preds, raw=False, agg_sim=True, agg_heads=agg_heads)

    def forward_batched(self, preds, agg_heads, batch_size):
        new_preds = {k: v for k, v in preds.items()}
        big_image_feats = new_preds.pop(IMAGE_FEATS)
        if self.use_cls:
            big_image_cls = new_preds.pop(IMAGE_CLS)

        n = big_image_feats.shape[0]
        n_steps = math.ceil(n / batch_size)
        outputs = []
        for step in tqdm(range(n_steps), "Calculating Sim", leave=False):
            new_preds[IMAGE_FEATS] = big_image_feats[
                step * batch_size : (step + 1) * batch_size
            ].cuda()
            if self.use_cls:
                new_preds[IMAGE_CLS] = big_image_cls[
                    step * batch_size : (step + 1) * batch_size
                ].cuda()

            sim = self.forward(new_preds, agg_heads=agg_heads)
            outputs.append(sim.cpu())
        return torch.cat(outputs, dim=1)
