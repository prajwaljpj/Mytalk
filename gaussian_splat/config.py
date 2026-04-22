from dataclasses import dataclass


@dataclass
class ModelConfig:
    path: str
    model_path: str = ""
    data_device: str = "cuda"
    white_background: bool = False
    eval: bool = False
    audio: str = ""
    init_num: int = 10_000
    audio_extractor: str = "ave"
    sh_degree: int = 2
    preload: bool = True
    cache_audio: bool = True

    @property
    def workspace(self) -> str:
        return self.model_path


@dataclass
class PipelineConfig:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False


@dataclass
class OptimizationConfig:
    iterations: int = 50_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 45_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.003
    rotation_lr: float = 0.001
    percent_dense: float = 0.005
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 45_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False
