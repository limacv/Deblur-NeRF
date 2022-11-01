# run script
# currently reso upsampling is not implemented
# --plenoxel: plenoxel option
# kernel_type: 'none' is vanilla 3d reconstruction, not DSK or CTK

python run_nerf.py --config configs/demo_blurball.txt --plenoxel --reso="[[256,256,128],[512,512,128]]" --kernel_type none --i_video 2000