
OPTIONS = dict()
OPTIONS["chunk_strategy"] = "nearest-gt"
OPTIONS["video_save_fps"] = 30.0
OPTIONS["beta_linear_start"] = 5e-6
OPTIONS["log_snr_shift"] = 2.4
OPTIONS["guider_types"] = 1
OPTIONS["camera_scale"] = 2.0
OPTIONS["num_steps"] = 50
OPTIONS["cfg"] = 2.0
OPTIONS["cfg_min"] = 1.2
OPTIONS["encoding_t"] = 1
OPTIONS["decoding_t"] = 1
OPTIONS["num_inputs"] = None
OPTIONS["seed"] = 1995

VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": OPTIONS,
}
