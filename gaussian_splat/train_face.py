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
from .motion_net import MotionNetwork
from .provider import GaussianScene
from .rich_utils import console, create_training_progress
from .renderer import render, render_motion
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
from .utils.sh_utils import eval_sh

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _loss_value(tensor):
    return float(tensor.detach().item())

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    testing_iterations = [i for i in range(0, opt.iterations + 1, 2000)]
    checkpoint_iterations =  saving_iterations = [i for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations]

    # vars
    warm_step = 3000
    opt.densify_until_iter = opt.iterations - 1000
    bg_iter = opt.iterations # opt.densify_until_iter
    lpips_start_iter = opt.densify_until_iter - 2000
    motion_stop_iter = bg_iter
    mouth_select_iter = bg_iter - 10000
    mouth_step = 1 / mouth_select_iter
    hair_mask_interval = 7
    select_interval = 15

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = GaussianScene(dataset, gaussians)

    motion_net = MotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / bg_iter))

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, motion_params, motion_optimizer_params, first_iter) = torch_load_unsafe(checkpoint)
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)

    bg_color = [0, 1, 0]   # [1, 1, 1] # if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1
    progress = create_training_progress(
        loss_columns=[
            "l1",
            "dssim",
            "motion_xyz",
            "motion_rot",
            "motion_opa",
            "motion_scale",
            "alpha",
            "attn_lips",
            "attn_hair_exp",
            "attn_hair_audio",
            "lpips_lips",
            "lpips_patch",
        ]
    )
    task_id = progress.add_task("Training", total=opt.iterations - first_iter + 1, status="Loss 0.00000")
    with progress:
        for iteration in range(first_iter, opt.iterations + 1):

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            mouth_global_lb = viewpoint_cam.talking_dict['mouth_bound'][0]
            mouth_global_ub = viewpoint_cam.talking_dict['mouth_bound'][1]
            mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2
            mouth_window = (mouth_global_ub - mouth_global_lb) * 0.2

            mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)
            mouth_ub = mouth_lb + mouth_window
            mouth_lb = mouth_lb - mouth_window

            au_global_lb = 0
            au_global_ub = 1
            au_window = 0.3

            au_lb = au_global_lb + mouth_step * iteration * (au_global_ub - au_global_lb)
            au_ub = au_lb + au_window
            au_lb = au_lb - au_window * 0.5

            if iteration < warm_step:
                if iteration % select_interval == 0:
                    while viewpoint_cam.talking_dict['mouth_bound'][2] < mouth_lb or viewpoint_cam.talking_dict['mouth_bound'][2] > mouth_ub:
                        if not viewpoint_stack:
                            viewpoint_stack = scene.getTrainCameras().copy()
                        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            if warm_step < iteration < mouth_select_iter:
                if iteration % select_interval == 0:
                    while viewpoint_cam.talking_dict['blink'] < au_lb or viewpoint_cam.talking_dict['blink'] > au_ub:
                        if not viewpoint_stack:
                            viewpoint_stack = scene.getTrainCameras().copy()
                        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            if viewpoint_cam.original_image == None:
                viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

            if (iteration - 1) == debug_from:
                pipe.debug = True

            face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
            hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
            mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
            head_mask =  face_mask + hair_mask

            if iteration > lpips_start_iter:
                max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                mouth_mask = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()

            hair_mask_iter = (warm_step < iteration < lpips_start_iter - 1000) and iteration % hair_mask_interval != 0

            if iteration < warm_step:
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            else:
                render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True)

            image_white, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image  = viewpoint_cam.original_image.cuda() / 255.0
            gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask

            if iteration > motion_stop_iter:
                for param in motion_net.parameters():
                    param.requires_grad = False
            if iteration > bg_iter:
                gaussians._xyz.requires_grad = False
                gaussians._opacity.requires_grad = False
                gaussians._scaling.requires_grad = False
                gaussians._rotation.requires_grad = False

            if iteration < bg_iter:
                if hair_mask_iter:
                    image_white[:, hair_mask] = background[:, None]
                    gt_image_white[:, hair_mask] = background[:, None]

                gt_image_white[:, mouth_mask] = background[:, None]

                Ll1 = l1_loss(image_white, gt_image_white)
                loss_l1 = Ll1
                loss_dssim = opt.lambda_dssim * (1.0 - ssim(image_white, gt_image_white))
                zero = torch.zeros([], device=gt_image.device)
                loss_motion_xyz = zero
                loss_motion_rot = zero
                loss_motion_opa = zero
                loss_motion_scale = zero
                loss_alpha = zero
                loss_attn_lips = zero
                loss_attn_hair_exp = zero
                loss_attn_hair_audio = zero
                loss_lpips_lips = zero
                loss_lpips_patch = zero

                if iteration > warm_step:
                    loss_motion_xyz = 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()
                    loss_motion_rot = 1e-5 * (render_pkg['motion']['d_rot'].abs()).mean()
                    loss_motion_opa = 1e-5 * (render_pkg['motion']['d_opa'].abs()).mean()
                    loss_motion_scale = 1e-5 * (render_pkg['motion']['d_scale'].abs()).mean()

                    loss_alpha = 1e-3 * (((1-alpha) * head_mask).mean() + (alpha * ~head_mask).mean())

                    [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
                    loss_attn_lips = 1e-4 * (render_pkg["attn"][1, xmin:xmax, ymin:ymax]).mean()
                    if not hair_mask_iter:
                        loss_attn_hair_exp = 1e-4 * (render_pkg["attn"][1][hair_mask]).mean()
                        loss_attn_hair_audio = 1e-4 * (render_pkg["attn"][0][hair_mask]).mean()

                loss = (
                    loss_l1
                    + loss_dssim
                    + loss_motion_xyz
                    + loss_motion_rot
                    + loss_motion_opa
                    + loss_motion_scale
                    + loss_alpha
                    + loss_attn_lips
                    + loss_attn_hair_exp
                    + loss_attn_hair_audio
                )

                image_t = image_white.clone()
                gt_image_t = gt_image_white.clone()

            else:
                image = image_white - background[:, None, None] * (1.0 - alpha) + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha)

                Ll1 = l1_loss(image, gt_image)
                loss_l1 = Ll1
                loss_dssim = opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                zero = torch.zeros([], device=gt_image.device)
                loss_motion_xyz = zero
                loss_motion_rot = zero
                loss_motion_opa = zero
                loss_motion_scale = zero
                loss_alpha = zero
                loss_attn_lips = zero
                loss_attn_hair_exp = zero
                loss_attn_hair_audio = zero
                loss_lpips_lips = zero
                loss_lpips_patch = zero
                loss = loss_l1 + loss_dssim

                image_t = image.clone()
                gt_image_t = gt_image.clone()

            if iteration > lpips_start_iter:
                [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
                pred_rgb = image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1
                rgb = gt_image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1

                padding_h = max(0, (32 - pred_rgb.shape[-2] + 1) // 2)
                padding_w = max(0, (32 - pred_rgb.shape[-1] + 1) // 2)

                if padding_w or padding_h:
                    pred_rgb = torch.nn.functional.pad(pred_rgb, (padding_w, padding_w, padding_h, padding_h))
                    rgb = torch.nn.functional.pad(rgb, (padding_w, padding_w, padding_h, padding_h))

                loss_lpips_lips = 0.01 * lpips_criterion(pred_rgb, rgb).mean()

                image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]
                gt_image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]

                patch_size = random.randint(32, 48) * 2
                loss_lpips_patch = 0.2 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()
                loss = loss + loss_lpips_lips + loss_lpips_patch

            loss_terms = {
                "l1": loss_l1,
                "dssim": loss_dssim,
                "motion_xyz": loss_motion_xyz,
                "motion_rot": loss_motion_rot,
                "motion_opa": loss_motion_opa,
                "motion_scale": loss_motion_scale,
                "alpha": loss_alpha,
                "attn_lips": loss_attn_lips,
                "attn_hair_exp": loss_attn_hair_exp,
                "attn_hair_audio": loss_attn_hair_audio,
                "lpips_lips": loss_lpips_lips,
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
                    progress.update(
                        task_id,
                        status=f"Loss={ema_loss_for_log:.5f} Mouth={mouth_lb:.1f}-{mouth_ub:.1f}",
                    )
                    progress.update_loss_table(iteration, loss.item(), loss_values)
                progress.advance(task_id)

                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, motion_net, render if iteration < warm_step else render_motion, (pipe, background))
                if iteration in saving_iterations:
                    console.log(f"[ITER {iteration}] Saving Gaussians")
                    scene.save(str(iteration) + '_face')

                if iteration in checkpoint_iterations:
                    console.log(f"[ITER {iteration}] Saving Checkpoint")
                    ckpt = (gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                    torch.save(ckpt, scene.model_path + "/chkpnt_face_" + str(iteration) + ".pth")
                    torch.save(ckpt, scene.model_path + "/chkpnt_face_latest" + ".pth")

                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                    dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                    bg_color_mask = (colors_precomp[..., 0] < 30/255) * (colors_precomp[..., 1] > 225/255) * (colors_precomp[..., 2] < 30/255)
                    gaussians.prune_points(bg_color_mask.squeeze())

                if iteration < opt.iterations:
                    motion_optimizer.step()
                    gaussians.optimizer.step()

                    motion_optimizer.zero_grad()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                    scheduler.step()



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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, motion_net, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 100, 5)]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if viewpoint.original_image == None:
                        viewpoint = loadCamOnTheFly(copy.deepcopy(viewpoint))
                        
                    if renderFunc is render:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    else:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, return_attn=True, frame_idx=0, *renderArgs)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha = render_pkg["alpha"]
                    image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + viewpoint.background.cuda() / 255.0 * (1.0 - alpha)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0)
                    
                    mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
                    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    mouth_mask_post = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), (render_pkg["depth"] / render_pkg["depth"].max())[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask_post".format(viewpoint.image_name), (~mouth_mask_post * gt_image)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask".format(viewpoint.image_name), (~mouth_mask[None] * gt_image)[None], global_step=iteration)

                        if renderFunc is not render:
                            tb_writer.add_images(config['name'] + "_view_{}/attn_a".format(viewpoint.image_name), (render_pkg["attn"][0] / render_pkg["attn"][0].max())[None, None], global_step=iteration)  
                            tb_writer.add_images(config['name'] + "_view_{}/attn_e".format(viewpoint.image_name), (render_pkg["attn"][1] / render_pkg["attn"][1].max())[None, None], global_step=iteration)  

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                console.log(f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    add_model_args(parser)
    add_optimization_args(parser)
    add_pipeline_args(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
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
