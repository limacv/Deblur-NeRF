
# [CREDIT]Plenoxel(cvpr2022)
# Most of them won't be used


import configargparse
from typing import Optional

def config_parser(parser:Optional[configargparse.ArgumentParser]=None):
    if parser is None:
        parser = configargparse.ArgumentParser()


    group = parser.add_argument_group("Data loading")
    group.add_argument('--scene_scale',
                         type=float,
                         default=None,
                         help="Global scene scaling (or use dataset default)")
    group.add_argument('--scale',
                         type=float,
                         default=None,
                         help="Image scale, e.g. 0.5 for half resolution (or use dataset default)")
    group.add_argument('--seq_id',
                         type=int,
                         default=1000,
                         help="Sequence ID (for CO3D only)")
    group.add_argument('--epoch_size',
                         type=int,
                         default=12800,
                         help="Pseudo-epoch size in term of batches (to be consistent across datasets)")
    # group.add_argument('--white_bkgd',
    #                      type=bool,
    #                      default=True,
    #                      help="Whether to use white background (ignored in some datasets)")
    # group.add_argument('--llffhold',
    #                      type=int,
    #                      default=8,
    #                      help="LLFF holdout every")
    group.add_argument('--normalize_by_bbox',
                         type=bool,
                         default=False,
                         help="Normalize by bounding box in bbox.txt, if available (NSVF dataset only); precedes normalize_by_camera")
    group.add_argument('--data_bbox_scale',
                         type=float,
                         default=1.2,
                         help="Data bbox scaling (NSVF dataset only)")
    group.add_argument('--cam_scale_factor',
                         type=float,
                         default=0.95,
                         help="Camera autoscale factor (NSVF/CO3D dataset only)")
    group.add_argument('--normalize_by_camera',
                         type=bool,
                         default=True,
                         help="Normalize using cameras, assuming a 360 capture (NSVF dataset only); only used if not normalize_by_bbox")
    group.add_argument('--perm', action='store_true', default=False,
                         help='sample by permutation of rays (true epoch) instead of '
                              'uniformly random rays')

    group = parser.add_argument_group("Render options")
    group.add_argument('--step_size',
                         type=float,
                         default=0.5,
                         help="Render step size (in voxel size units)")
    group.add_argument('--sigma_thresh',
                         type=float,
                         default=1e-8,
                         help="Skips voxels with sigma < this")
    group.add_argument('--stop_thresh',
                         type=float,
                         default=1e-7,
                         help="Ray march stopping threshold")
    group.add_argument('--background_brightness',
                         type=float,
                         default=1.0,
                         help="Brightness of the infinite background")
    group.add_argument('--renderer_backend', '-B',
                         choices=['cuvol', 'svox1', 'nvol'],
                         default='cuvol',
                         help="Renderer backend")
    group.add_argument('--random_sigma_std',
                         type=float,
                         default=0.0,
                         help="Random Gaussian std to add to density values (only if enable_random)")
    group.add_argument('--random_sigma_std_background',
                         type=float,
                         default=0.0,
                         help="Random Gaussian std to add to density values for BG (only if enable_random)")
    group.add_argument('--near_clip',
                         type=float,
                         default=0.00,
                         help="Near clip distance (in world space distance units, only for FG)")
    group.add_argument('--use_spheric_clip',
                         action='store_true',
                         default=False,
                         help="Use spheric ray clipping instead of voxel grid AABB "
                              "(only for FG; changes near_clip to mean 1-near_intersection_radius; "
                              "far intersection is always at radius 1)")
    group.add_argument('--enable_random',
                         action='store_true',
                         default=False,
                         help="Random Gaussian std to add to density values")
    group.add_argument('--last_sample_opaque',
                         action='store_true',
                         default=False,
                         help="Last sample has +1e9 density (used for LLFF)")
    

    group = parser.add_argument_group("general")
    group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                        help='checkpoint and logging directory')

    group.add_argument('--reso',
                            type=str,
                            default=
                            "[[256, 256, 256], [512, 512, 512]]",
                        help='List of grid resolution (will be evaled as json);'
                                'resamples to the next one every upsamp_every iters, then ' +
                                'stays at the last one; ' +
                                'should be a list where each item is a list of 3 ints or an int')
    group.add_argument('--upsamp_every', type=int, default=
                        3 * 12800,
                        help='upsample the grid every x iters')
    group.add_argument('--init_iters', type=int, default=
                        0,
                        help='do not upsample for first x iters')
    group.add_argument('--upsample_density_add', type=float, default=
                        0.0,
                        help='add the remaining density by this amount when upsampling')

    group.add_argument('--basis_type',
                        choices=['sh', '3d_texture', 'mlp'],
                        default='sh',
                        help='Basis function type')

    group.add_argument('--basis_reso', type=int, default=32,
                    help='basis grid resolution (only for learned texture)')
    group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

    group.add_argument('--mlp_posenc_size', type=int, default=4, help='Positional encoding size if using MLP basis; 0 to disable')
    group.add_argument('--mlp_width', type=int, default=32, help='MLP width if using MLP basis')

    group.add_argument('--background_nlayers', type=int, default=0,#32,
                    help='Number of background layers (0=disable BG model)')
    group.add_argument('--background_reso', type=int, default=512, help='Background resolution')



    group = parser.add_argument_group("optimization")
    group.add_argument('--n_iters', type=int, default=10 * 12800, help='total number of iters to optimize for')
    group.add_argument('--batch_size', type=int, default=
                        5000,
                        #100000,
                        #  2000,
                    help='batch size')


    # TODO: make the lr higher near the end
    group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
    group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
    group.add_argument('--lr_sigma_final', type=float, default=5e-2)
    group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)


    group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
    group.add_argument('--lr_sh', type=float, default=
                        1e-2,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_sh_final', type=float,
                        default=
                        5e-6
                        )
    group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

    group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

    # BG LRs
    group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
    group.add_argument('--lr_sigma_bg', type=float, default=3e0,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_sigma_bg_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sigma_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

    group.add_argument('--lr_color_bg', type=float, default=1e-1,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_color_bg_final', type=float, default=5e-6,#1e-4,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_color_bg_decay_steps', type=int, default=250000)
    group.add_argument('--lr_color_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
    # END BG LRs

    group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
    group.add_argument('--lr_basis', type=float, default=#2e6,
                        1e-6,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_basis_final', type=float,
                        default=
                        1e-6
                        )
    group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
    group.add_argument('--lr_basis_delay_steps', type=int, default=0,#15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_basis_begin_step', type=int, default=0)#4 * 12800)
    group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

    group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

    group.add_argument('--print_every', type=int, default=20, help='print every')
    group.add_argument('--save_every', type=int, default=5,
                    help='save every x epochs')
    group.add_argument('--eval_every', type=int, default=1,
                    help='evaluate every x epochs')

    group.add_argument('--init_sigma', type=float,
                    default=0.1,
                    help='initialization sigma')
    group.add_argument('--init_sigma_bg', type=float,
                    default=0.1,
                    help='initialization sigma (for BG)')

    # Extra logging
    group.add_argument('--log_mse_image', action='store_true', default=False)
    group.add_argument('--log_depth_map', action='store_true', default=False)
    group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
            help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


    group = parser.add_argument_group("misc experiments")
    group.add_argument('--thresh_type',
                        choices=["weight", "sigma"],
                        default="weight",
                    help='Upsample threshold type')
    group.add_argument('--weight_thresh', type=float,
                        default=0.0005 * 512,
                        #  default=0.025 * 512,
                    help='Upsample weight threshold; will be divided by resulting z-resolution')
    group.add_argument('--density_thresh', type=float,
                        default=5.0,
                    help='Upsample sigma threshold')
    group.add_argument('--background_density_thresh', type=float,
                        default=1.0+1e-9,
                    help='Background sigma threshold for sparsification')
    group.add_argument('--max_grid_elements', type=int,
                        default=44_000_000,
                    help='Max items to store after upsampling '
                            '(the number here is given for 22GB memory)')

    group.add_argument('--tune_mode', action='store_true', default=False,
                    help='hypertuning mode (do not save, for speed)')
    group.add_argument('--tune_nosave', action='store_true', default=False,
                    help='do not save any checkpoint even at the end')



    group = parser.add_argument_group("losses")
    # Foreground TV
    group.add_argument('--lambda_tv', type=float, default=1e-5)
    group.add_argument('--tv_sparsity', type=float, default=0.01)
    group.add_argument('--tv_logalpha', action='store_true', default=False,
                    help='Use log(1-exp(-delta * sigma)) as in neural volumes')

    group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
    group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

    group.add_argument('--lambda_tv_lumisphere', type=float, default=0.0)#1e-2)#1e-3)
    group.add_argument('--tv_lumisphere_sparsity', type=float, default=0.01)
    group.add_argument('--tv_lumisphere_dir_factor', type=float, default=0.0)

    group.add_argument('--tv_decay', type=float, default=1.0)

    group.add_argument('--lambda_l2_sh', type=float, default=0.0)#1e-4)
    group.add_argument('--tv_early_only', type=int, default=1, help="Turn off TV regularization after the first split/prune")

    group.add_argument('--tv_contiguous', type=int, default=1,
                            help="Apply TV only on contiguous link chunks, which is faster")
    # End Foreground TV

    group.add_argument('--lambda_sparsity', type=float, default=
                        0.0,
                        help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                            "(but applied on the ray)")
    group.add_argument('--lambda_beta', type=float, default=
                        0.0,
                        help="Weight for beta distribution sparsity loss as in neural volumes")


    # Background TV
    group.add_argument('--lambda_tv_background_sigma', type=float, default=1e-2)
    group.add_argument('--lambda_tv_background_color', type=float, default=1e-2)

    group.add_argument('--tv_background_sparsity', type=float, default=0.01)
    # End Background TV

    # Basis TV
    group.add_argument('--lambda_tv_basis', type=float, default=0.0,
                    help='Learned basis total variation loss')
    # End Basis TV

    group.add_argument('--weight_decay_sigma', type=float, default=1.0)
    group.add_argument('--weight_decay_sh', type=float, default=1.0)

    group.add_argument('--lr_decay', action='store_true', default=True)

    group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

    group.add_argument('--nosphereinit', action='store_true', default=False,
                        help='do not start with sphere bounds (please do not use for 360)')
    
    return parser