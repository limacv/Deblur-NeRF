import os
import numpy as np
import json
import cv2

data_path = "./blurtanabatabad"

# smart figure out path
input_path = os.path.join(data_path, 'raw')
json_path = os.path.join(input_path, "transforms.json")
out_path = os.path.join(data_path, "images_1")
pose_out_path = os.path.join(data_path, "poses_bounds.npy")
if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)
opengl2opencv = np.array([1, 0, 0, 0,
                          0, -1, 0, 0,
                          0, 0, -1, 0,
                          0, 0, 0, 1], dtype=np.float32).reshape(4,4)

# ================================================
# Blurxxx
# ================================================
if "blur" in data_path:
    with open(json_path, 'r') as metaf:
        meta = json.load(metaf)
        frames_data = meta["frames"]
        hold = meta["llffhold"]
        fov = meta["fov"]
        h, w = meta['h'], meta['w']
        cloest = meta['cloest']

    f = w / 2 / np.tan(fov / 2)
    hwf = np.array([h, w, f]).reshape([3, 1])
    pose_arr = []
    for i, frame_data in enumerate(frames_data):
        pose = frame_data["transform_matrix"]
        pose = np.array(pose, dtype=np.float32)
        pose = opengl2opencv @ np.linalg.inv(pose)
        pose = np.linalg.inv(pose)[:3]
        pose = np.concatenate([pose, hwf], 1)
        pose = np.concatenate([pose[:, 1:2], pose[:, 0:1], -pose[:, 2:3], pose[:, 3:4], pose[:, 4:5]],
                               1)
        pose = np.concatenate([pose.reshape(-1), np.array((cloest, cloest + 100), dtype=np.float32)])
        pose_arr.append(pose)
        assert (i % hold == 0) == (frame_data["blurcount"] == 0)  # valid check

        if i % hold == 0:
            img = cv2.imread(f"{input_path}/{i:03d}.png")
            cv2.imwrite(f"{out_path}/{i:03d}.png", img)
            print(f"frame {i} saved!")
        else:
            imgs = []
            for bluri in range(frame_data["blurcount"]):
                img = cv2.imread(f"{input_path}/{i:03d}_{bluri:03d}.png")
                imgs.append(img / 255)
            
            imgs = np.array(imgs)
            imgs = imgs ** 2.2
            weight = np.ones_like(imgs) * (1 / len(imgs))
            img = np.sum(imgs * weight, axis=0)
            img = img ** (1 / 2.2)
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(f"{out_path}/{i:03d}.png", img)
            print(f"frame {i} fused and saved!")
    
    pose_arr = np.stack(pose_arr)
    np.save(pose_out_path, pose_arr)
    print("poses_bounds.npy saved")

# ================================================
# Defocusxxx
# ================================================
else:
    with open(json_path, 'r') as metaf:
        meta = json.load(metaf)
        frames_data = meta["frames"]
        hold = meta["llffhold"]
        fov = meta["fov"]
        h, w = meta['h'], meta['w']
        cloest = meta['cloest']

    f = w / 2 / np.tan(fov / 2)
    hwf = np.array([h, w, f]).reshape([3, 1])
    pose_arr = []
    for i, frame_data in enumerate(frames_data):
        pose = frame_data["transform_matrix"]
        pose = np.array(pose, dtype=np.float32)
        pose = opengl2opencv @ np.linalg.inv(pose)
        pose = np.linalg.inv(pose)[:3]
        pose = np.concatenate([pose, hwf], 1)
        pose = np.concatenate([pose[:, 1:2], pose[:, 0:1], -pose[:, 2:3], pose[:, 3:4], pose[:, 4:5]],
                               1)
        pose = np.concatenate([pose.reshape(-1), np.array((cloest, cloest + 100), dtype=np.float32)])
        pose_arr.append(pose)

        img = cv2.imread(f"{input_path}/{i:03d}.png")
        cv2.imwrite(f"{out_path}/{i:03d}.png", img)
        print(f"frame {i} saved!")
    
    pose_arr = np.stack(pose_arr)
    np.save(pose_out_path, pose_arr)
    print("poses_bounds.npy saved")
