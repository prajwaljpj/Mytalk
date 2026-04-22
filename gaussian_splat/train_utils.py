import os
from argparse import ArgumentParser

from .config import ModelConfig, OptimizationConfig, PipelineConfig


def _add_bool_flag(parser: ArgumentParser, name: str, default: bool = False, help_text: str = ""):
    parser.add_argument(f"--{name}", action="store_true", default=default, help=help_text)


def add_model_args(parser: ArgumentParser):
    defaults = ModelConfig(path="")
    parser.add_argument(
        "--path",
        "--source_path",
        dest="path",
        required=True,
        help="Path to a MyTalk-preprocessed subject directory.",
    )
    parser.add_argument(
        "--workspace",
        "--model_path",
        dest="model_path",
        default="",
        help="Output directory for Gaussian checkpoints and point clouds.",
    )
    parser.add_argument("--audio", default="", help="Optional explicit audio feature or wav path.")
    parser.add_argument(
        "--audio_extractor",
        default="ave",
        choices=["ave", "deepspeech", "hubert", "esperanto"],
        help="Audio feature backend for the Gaussian motion networks.",
    )
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--init_num", type=int, default=10_000)
    parser.add_argument("--sh_degree", type=int, default=2)
    _add_bool_flag(parser, "white_background")
    _add_bool_flag(parser, "eval")
    parser.add_argument("--preload", dest="preload", action="store_true", default=defaults.preload)
    parser.add_argument("--no-preload", dest="preload", action="store_false")
    parser.add_argument(
        "--cache_audio",
        dest="cache_audio",
        action="store_true",
        default=defaults.cache_audio,
    )
    parser.add_argument("--no-cache_audio", dest="cache_audio", action="store_false")


def add_pipeline_args(parser: ArgumentParser):
    _add_bool_flag(parser, "convert_SHs_python")
    _add_bool_flag(parser, "compute_cov3D_python")
    _add_bool_flag(parser, "debug")


def add_optimization_args(parser: ArgumentParser):
    defaults = OptimizationConfig()
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--position_lr_init", type=float, default=defaults.position_lr_init)
    parser.add_argument("--position_lr_final", type=float, default=defaults.position_lr_final)
    parser.add_argument("--position_lr_delay_mult", type=float, default=defaults.position_lr_delay_mult)
    parser.add_argument("--position_lr_max_steps", type=int, default=defaults.position_lr_max_steps)
    parser.add_argument("--feature_lr", type=float, default=defaults.feature_lr)
    parser.add_argument("--opacity_lr", type=float, default=defaults.opacity_lr)
    parser.add_argument("--scaling_lr", type=float, default=defaults.scaling_lr)
    parser.add_argument("--rotation_lr", type=float, default=defaults.rotation_lr)
    parser.add_argument("--percent_dense", type=float, default=defaults.percent_dense)
    parser.add_argument("--lambda_dssim", type=float, default=defaults.lambda_dssim)
    parser.add_argument("--densification_interval", type=int, default=defaults.densification_interval)
    parser.add_argument("--opacity_reset_interval", type=int, default=defaults.opacity_reset_interval)
    parser.add_argument("--densify_from_iter", type=int, default=defaults.densify_from_iter)
    parser.add_argument("--densify_until_iter", type=int, default=defaults.densify_until_iter)
    parser.add_argument("--densify_grad_threshold", type=float, default=defaults.densify_grad_threshold)
    _add_bool_flag(parser, "random_background")


def extract_configs(args):
    model = ModelConfig(
        path=os.path.abspath(args.path),
        model_path=os.path.abspath(args.model_path) if args.model_path else "",
        data_device=args.data_device,
        white_background=args.white_background,
        eval=args.eval,
        audio=args.audio,
        init_num=args.init_num,
        audio_extractor=args.audio_extractor,
        sh_degree=args.sh_degree,
        preload=args.preload,
        cache_audio=args.cache_audio,
    )
    optimization = OptimizationConfig(
        iterations=args.iterations,
        position_lr_init=args.position_lr_init,
        position_lr_final=args.position_lr_final,
        position_lr_delay_mult=args.position_lr_delay_mult,
        position_lr_max_steps=args.position_lr_max_steps,
        feature_lr=args.feature_lr,
        opacity_lr=args.opacity_lr,
        scaling_lr=args.scaling_lr,
        rotation_lr=args.rotation_lr,
        percent_dense=args.percent_dense,
        lambda_dssim=args.lambda_dssim,
        densification_interval=args.densification_interval,
        opacity_reset_interval=args.opacity_reset_interval,
        densify_from_iter=args.densify_from_iter,
        densify_until_iter=args.densify_until_iter,
        densify_grad_threshold=args.densify_grad_threshold,
        random_background=args.random_background,
    )
    pipeline = PipelineConfig(
        convert_SHs_python=args.convert_SHs_python,
        compute_cov3D_python=args.compute_cov3D_python,
        debug=args.debug,
    )
    return model, optimization, pipeline
