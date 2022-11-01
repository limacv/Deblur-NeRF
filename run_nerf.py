
import os
import time

import cv2
import imageio
from tensorboardX import SummaryWriter

from NeRF import *
from load_llff import load_llff_data
from run_nerf_helpers import *
from metrics import compute_img_metric

import tqdm
import math

# np.random.seed(0)
DEBUG = False
##
import option_deblurnerf
import option_plenoxel

#"turns out that we don't need this"
#from alexyu_svox2_utils import convert_to_ndc

def train():
    parser = option_deblurnerf.config_parser()
    parser = option_plenoxel.config_parser(parser)

    args = parser.parse_args()
    
    
    if len(args.torch_hub_dir) > 0:
        print(f"Change torch hub cache to {args.torch_hub_dir}")
        torch.hub.set_dir(args.torch_hub_dir)

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args, args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify,
                                                                  path_epi=args.render_epi)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        print('LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.min(bds) * 0.9
            far = np.max(bds) * 1.0

        else:
            near = 0.
            far = 1.
        
        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    imagesf = images
    images = (images * 255).astype(np.uint8)
    images_idx = np.arange(0, len(images))

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses)

    # Create log dir and copy the config file
    basedir = args.basedir
    tensorboardbase = args.tbdir
    expname = args.expname
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(tensorboardbase, expname), exist_ok=True)

    tensorboard = SummaryWriter(os.path.join(tensorboardbase, expname))

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None and not args.render_only:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

        with open(test_metric_file, 'a') as file:
            file.write(open(args.config, 'r').read())
            file.write("\n============================\n"
                       "||\n"
                       "\\/\n")

    # The DSK module
    if args.kernel_type == 'deformablesparsekernel':
        kernelnet = DSKnet(len(images), torch.tensor(poses[:, :3, :4]),
                           args.kernel_ptnum, args.kernel_hwindow,
                           random_hwindow=args.kernel_random_hwindow, in_embed=args.kernel_rand_embed,
                           random_mode=args.kernel_random_mode,
                           img_embed=args.kernel_img_embed,
                           spatial_embed=args.kernel_spatial_embed,
                           depth_embed=args.kernel_depth_embed,
                           num_hidden=args.kernel_num_hidden,
                           num_wide=args.kernel_num_wide,
                           short_cut=args.kernel_shortcut,
                           pattern_init_radius=args.kernel_pattern_init_radius,
                           isglobal=args.kernel_isglobal,
                           optim_trans=args.kernel_global_trans,
                           optim_spatialvariant_trans=args.kernel_spatialvariant_trans)
    elif args.kernel_type == 'none':
        kernelnet = None
    else:
        raise RuntimeError(f"kernel_type {args.kernel_type} not recognized")

    # Create nerf model
    nerf = NeRFAll(args, kernelnet)
#    nerf = nn.DataParallel(nerf, list(range(args.num_gpu)))
    
    # If plenoxel, load plenoxel version lr scheduler
    if args.plenoxel:
        from alexyu_svox2_utils import get_expon_lr_func
        
        lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                        args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
        lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                                    args.lr_sh_delay_mult, args.lr_sh_decay_steps)
        lr_basis_func = get_expon_lr_func(args.lr_basis, args.lr_basis_final, args.lr_basis_delay_steps,
                                    args.lr_basis_delay_mult, args.lr_basis_decay_steps)
        lr_sigma_bg_func = get_expon_lr_func(args.lr_sigma_bg, args.lr_sigma_bg_final, args.lr_sigma_bg_delay_steps,
                                    args.lr_sigma_bg_delay_mult, args.lr_sigma_bg_decay_steps)
        lr_color_bg_func = get_expon_lr_func(args.lr_color_bg, args.lr_color_bg_final, args.lr_color_bg_delay_steps,
                                    args.lr_color_bg_delay_mult, args.lr_color_bg_decay_steps)
        lr_sigma_factor = 1.0
        lr_sh_factor = 1.0
        lr_basis_factor = 1.0

    else:
        
        optim_params = nerf.parameters()

        optimizer = torch.optim.Adam(params=optim_params,
                                    lr=args.lrate,
                                    betas=(0.9, 0.999))
    
    start = 0
    # Load Checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 '.tar' in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        if args.plenoxel:
            #checkpoitn mechanism for plenoxel
            #[TODO taekkii] implement here
            print("No implementation for plenoxel so far")
            print("Ignoring ckpt....")
        else:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # Load model
            smart_load_state_dict(nerf, ckpt)

    # figuring out the train/test configuration
    render_kwargs_train = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }
    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:  # args.dataset_type != 'llff' or
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    # visualize_motionposes(H, W, K, nerf, 2)
    # visualize_kernel(H, W, K, nerf, 5)
    # visualize_itsample(H, W, K, nerf)
    # visualize_kmap(H, W, K, nerf, img_idx=1)

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    global_step = start

    # Move testing data to GPU
    render_poses = torch.tensor(render_poses[:, :3, :4]).cuda()
    nerf = nerf.cuda()
    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname,
                                       f"renderonly"
                                       f"_{'test' if args.render_test else 'path'}"
                                       f"_{start:06d}")
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            nerf.eval()
            rgbshdr, disps = nerf(
                hwf[0], hwf[1], K, args.chunk,
                poses=torch.cat([render_poses, dummy_poses], dim=0),
                render_kwargs=render_kwargs_test,
                render_factor=args.render_factor,
            )
            rgbshdr = rgbshdr[:len(rgbshdr) - dummy_num]
            disps = (1. - disps)
            disps = disps[:len(disps) - dummy_num].cpu().numpy()
            rgbs = rgbshdr
            rgbs = to8b(rgbs.cpu().numpy())
            disps = to8b(disps / disps.max())
            if args.render_test:
                for rgb_idx, rgb8 in enumerate(rgbs):
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}.png'), rgb8)
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_disp.png'), disps[rgb_idx])
            else:
                prefix = 'epi_' if args.render_epi else ''
                imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video.mp4'), rgbs, fps=30, quality=9)
                imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video_disp.mp4'), disps, fps=30, quality=9)

            if args.render_test and args.render_multipoints:
                for pti in range(args.kernel_ptnum):
                    nerf.eval()
                    poses_num = len(poses) + dummy_num
                    imgidx = torch.arange(poses_num, dtype=torch.long).to(render_poses.device).reshape(poses_num, 1)
                    rgbs, weights = nerf(
                        hwf[0], hwf[1], K, args.chunk,
                        poses=torch.cat([render_poses, dummy_poses], dim=0),
                        render_kwargs=render_kwargs_test,
                        render_factor=args.render_factor,
                        render_point=pti,
                        images_indices=imgidx
                    )
                    rgbs = rgbs[:len(rgbs) - dummy_num]
                    weights = weights[:len(weights) - dummy_num]
                    rgbs = to8b(rgbs.cpu().numpy())
                    weights = to8b(weights.cpu().numpy())
                    for rgb_idx, rgb8 in enumerate(rgbs):
                        imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_pt{pti}.png'), rgb8)
                        imageio.imwrite(os.path.join(testsavedir, f'w_{rgb_idx:03d}_pt{pti}.png'), weights[rgb_idx])
            return

    # ============================================
    # Prepare ray dataset if batching random rays
    # ============================================
    N_rand = args.N_rand
    train_datas = {}
    
    # if downsample, downsample the images
    if args.datadownsample > 0:
        images_train = np.stack([cv2.resize(img_, None, None,
                                            1 / args.datadownsample, 1 / args.datadownsample,
                                            cv2.INTER_AREA) for img_ in imagesf], axis=0)
    else:
        images_train = imagesf

    num_img, hei, wid, _ = images_train.shape
    print(f"train on image sequence of len = {num_img}, {wid}x{hei}")
    k_train = np.array([K[0, 0] * wid / W, 0, K[0, 2] * wid / W,
                        0, K[1, 1] * hei / H, K[1, 2] * hei / H,
                        0, 0, 1]).reshape(3, 3).astype(K.dtype)

    # For random ray batching
    print('get rays')
    rays = np.stack([get_rays_np(hei, wid, k_train, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
    rays = np.transpose(rays, [0, 2, 3, 1, 4])
    train_datas['rays'] = rays[i_train].reshape(-1, 2, 3)
    
    
        
    xs, ys = np.meshgrid(np.arange(wid, dtype=np.float32), np.arange(hei, dtype=np.float32), indexing='xy')
    xs = np.tile((xs[None, ...] + HALF_PIX) * W / wid, [num_img, 1, 1])
    ys = np.tile((ys[None, ...] + HALF_PIX) * H / hei, [num_img, 1, 1])
    train_datas['rays_x'], train_datas['rays_y'] = xs[i_train].reshape(-1, 1), ys[i_train].reshape(-1, 1)

    train_datas['rgbsf'] = images_train[i_train].reshape(-1, 3)

    images_idx_tile = images_idx.reshape((num_img, 1, 1))
    images_idx_tile = np.tile(images_idx_tile, [1, hei, wid])
    train_datas['images_idx'] = images_idx_tile[i_train].reshape(-1, 1).astype(np.int64)

    print('shuffle rays')
    shuffle_idx = np.random.permutation(len(train_datas['rays']))
    train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}

    print('done')
    i_batch = 0

    # Move training data to GPU
    images = torch.tensor(images).cuda()
    imagesf = torch.tensor(imagesf).cuda()

    poses = torch.tensor(poses).cuda()
    train_datas = {k: torch.tensor(v).cuda() for k, v in train_datas.items()}

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in tqdm.tqdm(range(start, N_iters)):
        time0 = time.time()

        # Sample random ray batch
        iter_data = {k: v[i_batch:i_batch + N_rand] for k, v in train_datas.items()}
        batch_rays = iter_data.pop('rays').permute(0, 2, 1)

        i_batch += N_rand
        if i_batch >= len(train_datas['rays']):
            print("Shuffle data after an epoch!")
            shuffle_idx = np.random.permutation(len(train_datas['rays']))
            train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}
            i_batch = 0

        #####  Core optimization loop  #####
        nerf.train()
        if i == args.kernel_start_iter:
            torch.cuda.empty_cache()
        
        target_rgb = iter_data['rgbsf'].squeeze(-2)
        
        rgb, rgb0, extra_loss = nerf(H, W, K, chunk=args.chunk,
                                    rays=batch_rays, rays_info=iter_data,
                                    retraw=True, force_naive=i < args.kernel_start_iter,gt=target_rgb,
                                    **render_kwargs_train)
        
        # Compute Losses
        # =====================
        

        if args.plenoxel:
            #copy-n-paste! [taekkii TODO] re-organize this
            
            #run lr-schedulers
            plenoxel_model = nerf.plenoxel

 
            if args.lr_fg_begin_step > 0 and i == args.lr_fg_begin_step:
                plenoxel_model.density_data.data[:] = args.init_sigma
            lr_sigma = lr_sigma_func(i) * lr_sigma_factor
            lr_sh = lr_sh_func(i) * lr_sh_factor
            lr_basis = lr_basis_func(i - args.lr_basis_begin_step) * lr_basis_factor
            lr_sigma_bg = lr_sigma_bg_func(i - args.lr_basis_begin_step) * lr_basis_factor
            lr_color_bg = lr_color_bg_func(i - args.lr_basis_begin_step) * lr_basis_factor
            if not args.lr_decay:
                lr_sigma = args.lr_sigma * lr_sigma_factor
                lr_sh = args.lr_sh * lr_sh_factor
                lr_basis = args.lr_basis * lr_basis_factor
            
            
            mse = F.mse_loss(rgb, target_rgb)

            # Stats
            mse_num : float = mse.detach().item()
           
            psnr = -10.0 * math.log10(mse_num)
            if i % 5000==0:
                import pdb;pdb.set_trace()
            # Apply TV/Sparsity regularizers
            # if args.lambda_tv > 0.0:
            #     #  with Timing("tv_inpl"):
            #     plenoxel_model.inplace_tv_grad(plenoxel_model.density_data.grad,
            #             scaling=args.lambda_tv,
            #             sparse_frac=args.tv_sparsity,
            #             logalpha=args.tv_logalpha,
            #             ndc_coeffs=nerf.ndc_coeffs,
            #             contiguous=args.tv_contiguous)
            # if args.lambda_tv_sh > 0.0:
            #     #  with Timing("tv_color_inpl"):
            #     plenoxel_model.inplace_tv_color_grad(plenoxel_model.sh_data.grad,
            #             scaling=args.lambda_tv_sh,
            #             sparse_frac=args.tv_sh_sparsity,
            #             ndc_coeffs=nerf.ndc_coeffs,
            #             contiguous=args.tv_contiguous)
            # if args.lambda_tv_lumisphere > 0.0:
            #     plenoxel_model.inplace_tv_lumisphere_grad(plenoxel_model.sh_data.grad,
            #             scaling=args.lambda_tv_lumisphere,
            #             dir_factor=args.tv_lumisphere_dir_factor,
            #             sparse_frac=args.tv_lumisphere_sparsity,
            #             ndc_coeffs=nerf.ndc_coeffs)
            # if args.lambda_l2_sh > 0.0:
            #     plenoxel_model.inplace_l2_color_grad(plenoxel_model.sh_data.grad,
            #             scaling=args.lambda_l2_sh)
            # if plenoxel_model.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
            #     plenoxel_model.inplace_tv_background_grad(plenoxel_model.background_data.grad,
            #             scaling=args.lambda_tv_background_color,
            #             scaling_density=args.lambda_tv_background_sigma,
            #             sparse_frac=args.tv_background_sparsity,
            #             contiguous=args.tv_contiguous)
            # if args.lambda_tv_basis > 0.0:
            #     tv_basis = plenoxel_model.tv_basis()
            #     loss_tv_basis = tv_basis * args.lambda_tv_basis
            #     loss_tv_basis.backward()
            #  print('nz density', torch.count_nonzero(grid.sparse_grad_indexer).item(),
            #        ' sh', torch.count_nonzero(grid.sparse_sh_grad_indexer).item())


            # run plenoxel optimizers
            # Manual SGD/rmsprop step
            if i >= args.lr_fg_begin_step:
               
                plenoxel_model.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
                plenoxel_model.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
            if plenoxel_model.use_background:
                plenoxel_model.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)
            if i >= args.lr_basis_begin_step:
                if plenoxel_model.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    plenoxel_model.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
                # elif nerf.basis_type == svox2.BASIS_TYPE_MLP:
                #     optim_basis_mlp.step()
                #     optim_basis_mlp.zero_grad()
        else:
            img_loss = img2mse(rgb, target_rgb)
            loss = img_loss
            psnr = mse2psnr(img_loss)
            if rgb0 is not None:
                img_loss0 = img2mse(rgb0, target_rgb)
                loss = loss + img_loss0

            extra_loss = {k: torch.mean(v) for k, v in extra_loss.items()}
            if len(extra_loss) > 0:
                for k, v in extra_loss.items():
                    if f"kernel_{k}_weight" in vars(args).keys():
                        if vars(args)[f"{k}_start_iter"] <= i <= vars(args)[f"{k}_end_iter"]:
                            loss = loss + v * vars(args)[f"kernel_{k}_weight"]
        
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

        # dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.plenoxel:
                torch.save({
                    'global_step': global_step,
                    'network_state_dict': nerf.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_state_dict': nerf.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            #[taekkii]VIDEO RENDERING IS NOT IMPLEMENTED YET
            # Turn on testing mode
            #import pdb;pdb.set_trace()
            with torch.no_grad():
                nerf.eval()
                rgbs, disps = nerf(H, W, K, args.chunk, poses=render_poses, render_kwargs=render_kwargs_test)
                rgbs = rgbs.reshape(-1,H,W,3)
            
            #print('Done, saving', rgbs.shape, disps.shape)
            print("done")
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            rgbs = (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
            rgbs = rgbs.cpu().numpy()
            
            if not args.plenoxel:
                disps = disps.cpu().numpy()
            # disps_max_idx = int(disps.size * 0.9)
            # disps_max = disps.reshape(-1)[np.argpartition(disps.reshape(-1), disps_max_idx)[disps_max_idx]]

            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            #temporarily disables disp map for plenoxel
            if not args.plenoxel:
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / disps.max()), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses.shape)
            dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            with torch.no_grad():
                nerf.eval()
                rgbs, _ = nerf(H, W, K, args.chunk, poses=torch.cat([poses, dummy_poses], dim=0).cuda(),
                               render_kwargs=render_kwargs_test)
                rgbs = rgbs[:len(rgbs) - dummy_num]
                rgbs_save = rgbs  # (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
                # saving
                for rgb_idx, rgb in enumerate(rgbs_save):
                    rgb8 = to8b(rgb.cpu().numpy())
                    filename = os.path.join(testsavedir, f'{rgb_idx:03d}.png')
                    imageio.imwrite(filename, rgb8)

                # evaluation
                rgbs = rgbs[i_test]
                target_rgb_ldr = imagesf[i_test]

                test_mse = compute_img_metric(rgbs, target_rgb_ldr, 'mse')
                test_psnr = compute_img_metric(rgbs, target_rgb_ldr, 'psnr')
                test_ssim = compute_img_metric(rgbs, target_rgb_ldr, 'ssim')
                test_lpips = compute_img_metric(rgbs, target_rgb_ldr, 'lpips')
                if isinstance(test_lpips, torch.Tensor):
                    test_lpips = test_lpips.item()

                tensorboard.add_scalar("Test MSE", test_mse, global_step)
                tensorboard.add_scalar("Test PSNR", test_psnr, global_step)
                tensorboard.add_scalar("Test SSIM", test_ssim, global_step)
                tensorboard.add_scalar("Test LPIPS", test_lpips, global_step)

            with open(test_metric_file, 'a') as outfile:
                outfile.write(f"iter{i}/globalstep{global_step}: MSE:{test_mse:.8f} PSNR:{test_psnr:.8f}"
                              f" SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}\n")

            print('Saved test set')

        if i % args.i_tensorboard == 0 :
            #[TODO] Implment here for plenoxel

            tensorboard.add_scalar("Loss", mse_num if args.plenoxel else loss.item(), global_step)
            tensorboard.add_scalar("PSNR", psnr    if args.plenoxel else psnr.item(), global_step)
            for k, v in extra_loss.items():
                tensorboard.add_scalar(k, v.item(), global_step)

        if i % args.i_print == 0:
            print(f"[TRAIN] Iter: {i} Loss: {mse_num if args.plenoxel else loss.item():.4f}  PSNR: {psnr if args.plenoxel else psnr.item():.5f}")
            if args.plenoxel:
                print(f"        lr_sigma: {lr_sigma:.4f} lr_sh: {lr_sh:.4f}")

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
