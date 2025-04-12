import torch
import threading
import gradio as gr

from .sampling import EulerEDMSampler, MultiviewCFG, MultiviewTemporalCFG, VanillaCFG


class GradioTrackedSampler(EulerEDMSampler):
    """
    A thin wrapper around the EulerEDMSampler that allows tracking progress and
    aborting sampling for gradio demo.
    """

    def __init__(self, abort_event: threading.Event, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abort_event = abort_event

    # type: ignore
    def __call__(
        self,
        denoiser,
        x: torch.Tensor,
        scale: float | torch.Tensor,
        cond: dict,
        uc: dict | None = None,
        num_steps: int | None = None,
        verbose: bool = True,
        global_pbar: gr.Progress | None = None,
        **guider_kwargs,
    ) -> torch.Tensor | None:

        uc = cond if uc is None else uc
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas, verbose=verbose):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )

            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i+1],
                denoiser, x, scale, cond, uc, gamma, **guider_kwargs,
            )

            # Allow tracking progress in gradio demo.
            if global_pbar is not None:
                global_pbar.update()

            # Allow aborting sampling in gradio demo.
            if self.abort_event.is_set():
                return None

        return x


def create_samplers(
    guider_types: int | list[int],
    discretization,
    num_frames: list[int] | None,
    num_steps: int,
    cfg_min: float = 1.0,
    device: str | torch.device = "cuda",
    abort_event: threading.Event | None = None,
):
    guider_mapping = {
        0: VanillaCFG,
        1: MultiviewCFG,
        2: MultiviewTemporalCFG,
    }
    samplers = []
    if not isinstance(guider_types, (list, tuple)):
        guider_types = [guider_types]
    for i, guider_type in enumerate(guider_types):
        if guider_type not in guider_mapping:
            raise ValueError(
                f"Invalid guider type {guider_type}. Must be one of {list(guider_mapping.keys())}"
            )
        guider_cls = guider_mapping[guider_type]
        guider_args = ()
        if guider_type > 0:
            guider_args += (cfg_min,)
            if guider_type == 2:
                assert num_frames is not None
                guider_args = (num_frames[i], cfg_min)
        guider = guider_cls(*guider_args)

        if abort_event is not None:
            sampler = GradioTrackedSampler(
                abort_event,
                discretization=discretization,
                guider=guider,
                num_steps=num_steps,
                s_churn=0.0,
                s_tmin=0.0,
                s_tmax=999.0,
                s_noise=1.0,
                verbose=True,
                device=device,
            )
        else:
            sampler = EulerEDMSampler(
                discretization=discretization,
                guider=guider,
                num_steps=num_steps,
                s_churn=0.0,
                s_tmin=0.0,
                s_tmax=999.0,
                s_noise=1.0,
                verbose=True,
                device=device,
            )
        samplers.append(sampler)
    return samplers


def do_sample(
    model,
    ae,
    conditioner,
    denoiser,
    sampler,
    value_dict,
    H,
    W,
    C,
    F,
    T,
    cfg,
    encoding_t=1,
    decoding_t=1,
    verbose=True,
    global_pbar=None,
    **_,
):
    imgs = value_dict["cond_frames"].to("cuda")
    input_masks = value_dict["cond_frames_mask"].to("cuda")
    pluckers = value_dict["plucker_coordinate"].to("cuda")

    num_samples = [1, T]
    with torch.inference_mode(), torch.autocast("cuda"):
        load_model(ae)
        load_model(conditioner)
        latents = torch.nn.functional.pad(
            ae.encode(imgs[input_masks], encoding_t), (0, 0, 0, 0, 0, 1), value=1.0
        )
        c_crossattn = repeat(conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T)
        uc_crossattn = torch.zeros_like(c_crossattn)
        c_replace = latents.new_zeros(T, *latents.shape[1:])
        c_replace[input_masks] = latents
        uc_replace = torch.zeros_like(c_replace)
        c_concat = torch.cat(
            [
                repeat(
                    input_masks,
                    "n -> n 1 h w",
                    h=pluckers.shape[2],
                    w=pluckers.shape[3],
                ),
                pluckers,
            ],
            1,
        )
        uc_concat = torch.cat(
            [pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers], 1
        )
        c_dense_vector = pluckers
        uc_dense_vector = c_dense_vector
        c = {
            "crossattn": c_crossattn,
            "replace": c_replace,
            "concat": c_concat,
            "dense_vector": c_dense_vector,
        }
        uc = {
            "crossattn": uc_crossattn,
            "replace": uc_replace,
            "concat": uc_concat,
            "dense_vector": uc_dense_vector,
        }
        unload_model(ae)
        unload_model(conditioner)

        additional_model_inputs = {"num_frames": T}
        additional_sampler_inputs = {
            "c2w": value_dict["c2w"].to("cuda"),
            "K": value_dict["K"].to("cuda"),
            "input_frame_mask": value_dict["cond_frames_mask"].to("cuda"),
        }
        if global_pbar is not None:
            additional_sampler_inputs["global_pbar"] = global_pbar

        shape = (math.prod(num_samples), C, H // F, W // F)
        randn = torch.randn(shape).to("cuda")

        load_model(model)
        samples_z = sampler(
            lambda input, sigma, c: denoiser(
                model,
                input,
                sigma,
                c,
                **additional_model_inputs,
            ),
            randn,
            scale=cfg,
            cond=c,
            uc=uc,
            verbose=verbose,
            **additional_sampler_inputs,
        )
        if samples_z is None:
            return
        unload_model(model)

        load_model(ae)
        samples = ae.decode(samples_z, decoding_t)
        unload_model(ae)

    return samples

