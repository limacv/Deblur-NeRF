


import numpy as np
import torch
def convert_to_ndc(origins, directions, ndc_coeffs, near: float = 1.0):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane, not sure if needed
    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)

    # Projection
    o0 = ndc_coeffs[0] * (ox / oz)
    o1 = ndc_coeffs[1] * (oy / oz)
    o2 = 1 - 2 * near / oz

    d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    d2 = 2 * near / oz;

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)
    return origins, directions

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper