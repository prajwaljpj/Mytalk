#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
import copy
from random import randint

import lpips
import torch

from .gaussian_model import GaussianModel
from .motion_net import MotionNetwork, MouthMotionNetwork
from .provider import GaussianScene
from .rich_utils import console, create_training_progress
from .renderer import render, render_motion, render_motion_mouth
from .train_utils import (
    add_model_args,
    add_optimization_args,
    add_pipeline_args,
    extract_configs,
)
from .utils.camera_utils import loadCamOnTheFly
from .utils.general_utils import safe_state, torch_load_unsafe
from .utils.image_utils import psnr
from .utils.loss_utils import l1_loss, l2_loss, patchify, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _loss_value(tensor):
    return float(tensor.detach().item())

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    opt.iterations = 10000
    opt.densify_until_iter = 0

    testing_iterations = [i for i in range(0, opt.iterations + 1, 2000)]
    checkpoint_iterations = [opt.iterations]

    # vars
    bg_iter = opt.densify_until_iter
    lpips_start_iter = 5000

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = GaussianScene(dataset, gaussians)
    gaussians_mouth = GaussianModel(dataset.sh_degree)
    with torch.no_grad():
        motion_net_mouth = MouthMotionNetwork(args=dataset).cuda()
        motion_net = MotionNetwork(args=dataset).cuda()

    gaussians.training_setup(opt)
    gaussians_mouth.training_setup(opt)

    (model_params, motion_params, _, _) = torch_load_unsafe(os.path.join(scene.model_path, "chkpnt_face_latest.pth"))
    gaussians.restore(model_params, opt)
    motion_net.load_state_dict(motion_params)

    (model_params, motion_params, _, _) = torch_load_unsafe(os.path.join(scene.model_path, "chkpnt_mouth_latest.pth"))
    gaussians_mouth.restore(model_params, opt)
    motion_net_mouth.load_state_dict(motion_params)

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()

    bg_color = [0, 1, 0]   # [1, 1, 1] # if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1
    progress = create_training_progress(
        loss_columns=["l1", "dssim", "alpha", "lpips_patch"]
    )
    task_id = progress.add_task("Training", total=opt.iterations - first_iter + 1, status="Loss 0.00000")

    with progress:
        for iteration in range(first_iter, opt.iterations + 1):

            iter_start.record()

            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            if viewpoint_cam.original_image == None:
                viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

            gaussians.update_learning_rate(iteration)

            face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
            hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
            mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
            head_mask = face_mask + hair_mask + mouth_mask

            if (iteration - 1) == debug_from:
                pipe.debug = True

            render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background)
            render_pkg_mouth = render_motion_mouth(viewpoint_cam, gaussians_mouth, motion_net_mouth, pipe, background)
            viewspace_point_tensor, visibility_filter = render_pkg["viewspace_points"], render_pkg["visibility_filter"]
            viewspace_point_tensor_mouth, visibility_filter_mouth = render_pkg_mouth["viewspace_points"], render_pkg_mouth["visibility_filter"]

            alpha_mouth = render_pkg_mouth["alpha"]
            alpha = render_pkg["alpha"]
            mouth_image = render_pkg_mouth["render"] - background[:, None, None] * (1.0 - alpha_mouth) + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha_mouth)
            image = render_pkg["render"] - background[:, None, None] * (1.0 - alpha) + mouth_image * (1.0 - alpha)

            gt_image  = viewpoint_cam.original_image.cuda() / 255.0
            gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask

            if iteration > bg_iter:
                for param in motion_net.parameters():
                    param.requires_grad = False
                for param in motion_net_mouth.parameters():
                    param.requires_grad = False

                gaussians._xyz.requires_grad = False
                gaussians._scaling.requires_grad = False
                gaussians._rotation.requires_grad = False

                gaussians_mouth._xyz.requires_grad = False
                gaussians_mouth._opacity.requires_grad = False
                gaussians_mouth._scaling.requires_grad = False
                gaussians_mouth._rotation.requires_grad = False

            if iteration < bg_iter:
                image[:, ~head_mask] = background[:, None]

                Ll1 = l1_loss(image, gt_image_white)
                loss_l1 = Ll1
                loss_dssim = opt.lambda_dssim * (1.0 - ssim(image, gt_image_white))
                loss_alpha = 1e-3 * (((1-alpha) * head_mask).mean() + (alpha * ~head_mask).mean())
                loss_lpips_patch = torch.zeros([], device=gt_image.device)
                loss = loss_l1 + loss_dssim + loss_alpha

                image_t = image.clone()
                gt_image_t = gt_image_white.clone()

            else:
                Ll1 = l1_loss(image, gt_image)
                loss_l1 = Ll1
                loss_dssim = opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                loss_alpha = torch.zeros([], device=gt_image.device)
                loss_lpips_patch = torch.zeros([], device=gt_image.device)
                loss = loss_l1 + loss_dssim

                image_t = image.clone()
                gt_image_t = gt_image.clone()

            if iteration > lpips_start_iter:
                patch_size = random.randint(16, 21) * 2
                loss_lpips_patch = 0.5 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()
                loss = loss + loss_lpips_patch

            loss_terms = {
                "l1": loss_l1,
                "dssim": loss_dssim,
                "alpha": loss_alpha,
                "lpips_patch": loss_lpips_patch,
            }

            loss.backward()

            iter_end.record()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    loss_values = {
                        name: _loss_value(value) for name, value in loss_terms.items()
                    }
                    progress.update(task_id, status=f"Loss={ema_loss_for_log:.5f}")
                    progress.update_loss_table(iteration, loss.item(), loss_values)
                progress.advance(task_id)

                training_report(tb_writer, iteration, testing_iterations, image, gt_image)
                if iteration in saving_iterations:
                    console.log(f"[ITER {iteration}] Saving Gaussians")
                    scene.save(iteration)

                if iteration in checkpoint_iterations:
                    console.log(f"[ITER {iteration}] Saving Checkpoint")
                    ckpt = (gaussians.capture(), motion_net.state_dict(), gaussians_mouth.capture(), motion_net_mouth.state_dict())
                    torch.save(ckpt, scene.model_path + "/chkpnt_fuse_" + str(iteration) + ".pth")
                    torch.save(ckpt, scene.model_path + "/chkpnt_fuse_latest" + ".pth")

                if iteration < opt.densify_until_iter:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    gaussians_mouth.add_densification_stats(viewspace_point_tensor_mouth, visibility_filter_mouth)

                    if iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.3, scene.cameras_extent, size_threshold)
                        gaussians_mouth.densify_and_prune(opt.densify_grad_threshold, 0.3, scene.cameras_extent, size_threshold)

                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians_mouth.optimizer.step()

                    gaussians.optimizer.zero_grad(set_to_none = True)
                    gaussians_mouth.optimizer.zero_grad(set_to_none = True)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    console.log(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        console.log("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, testing_iterations, image, gt_image):
    # Report test and samples of training set
    if tb_writer and iteration in testing_iterations:
        tb_writer.add_images("fuse/render", image[None], global_step=iteration)
        tb_writer.add_images("fuse/ground_truth", gt_image[None], global_step=iteration)



if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    add_model_args(parser)
    add_optimization_args(parser)
    add_pipeline_args(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset, opt, pipe = extract_configs(args)
    console.log(f"Optimizing {dataset.model_path or '<auto>'}")
    training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    console.log("Training complete.")
