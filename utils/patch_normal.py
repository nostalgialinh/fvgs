


import os
import collections
import numpy as np
import torch
from PIL import Image as pil_image
import math



def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
class Point3D:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    def __init__(self) -> None:
        self.id: int = 0
        self.xyz: np.ndarray = np.zeros(3)
        self.rgb: np.ndarray = np.zeros(3)
        self.error: float = 0.0
        self.track: list[tuple] = []

class Image:
    img_patch_size = 336
    img8_patch_size = 42
    img_size = (4032,3024) #(width, height)
    img8_size = (504,378)

    def __init__(self, id: int, points2d: list[(float,float, int)], name: str, qvec: np.array, tvec: np.array, camera_id: int) -> None:
        self.id = id
        self.points2d = points2d
        self.image_name = name
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.used_keypoints_idx: list[(int,int, np.array[float,float,float])] = [] # (point2d_idx, point3d_id, xyz in 3d)
        self.depth_map_pcd: np.array = None
        self.camera_intrinsics = None

    def create_depth_map_pcd(self):
        x: int = None
        y: int = None
        focal_length_x = self.camera_intrinsics.params[0]
        focal_length_y = self.camera_intrinsics.params[0]
        height = self.img_size[1]
        width = self.img_size[0]
        self.depth_map_pcd = np.full((self.img_size[1], self.img_size[0]), np.inf)
        K = np.array([[focal_length_x, 0, width/2], [0, focal_length_y, height/2], [0, 0, 1]])
        R = np.transpose(qvec2rotmat(self.qvec))
        T = np.array(self.tvec)
        for elem in self.used_keypoints_idx:
            idx = elem[0]
            x = int(self.points2d[idx][0])
            y = int(self.points2d[idx][1])
            point_3d_xyz = np.vstack(elem[2]) # np array xyz
            cam_coord = np.matmul(K, np.matmul(R.transpose(), point_3d_xyz) + T.reshape(3,1))
            # print(np.shape(point_3d_xyz))
            depth = cam_coord[2]
            # print(np.shape(depth))
            if self.depth_map_pcd[y,x] > depth:
                self.depth_map_pcd[y,x] = depth



Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])     
def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                # assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_point3d(path):
    points: list[Point3D] = []
    with open(path, 'r') as f:
        for _ in range(3):
            f.readline()
        # count = 0
        while True:
            line = f.readline()
            # count+=1
            if not line:
                break
            tokens = line.split()
            point3d = Point3D()
            point3d.id = (int(tokens[0]))
            point3d.xyz = np.array(tuple(map(float, tokens[1:4])))
            # if count == 1:
            #     print('HEHE')
            #     print(np.shape(point3d.xyz)) 
            if point3d.xyz[2] < 0:
                print("Exists z smaller than 0")
            point3d.rgb = np.array([float(tokens[4]), float(tokens[5]), float(tokens[6])])
            point3d.error = float(tokens[7])
            for i in range(8, len(tokens), 2):
                image_id = int(tokens[i])
                point2d_idx = int(tokens[i+1])
                point3d.track.append((image_id, point2d_idx))
            points.append(point3d)
    return points

def read_images(path):
    images = []
    with open(path, 'r') as f:
        for _ in range(4):
            f.readline()
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        while True:
            line = f.readline()
            if not line:
                break
            tokens = line.split()
            image_id = int(tokens[0])
            qvec = np.array(tuple(map(float, tokens[1:5])))
            tvec = np.array(tuple(map(float, tokens[5:8])))
            camera_id = int(tokens[8])
            image_name = tokens[9]

            #read points2d
            line = f.readline()
            tokens = line.split()
            points2d = []
            for i in range(0, len(tokens), 3):
                x = float(tokens[i])
                y = float(tokens[i+1])
                point3d_id = int(tokens[i+2])
                points2d.append((x, y, point3d_id))
            images.append(Image(image_id, points2d, image_name, qvec, tvec, camera_id))
    images.sort(key=lambda x: x.id)
    return images

def set_used_keypoints(images, points3D):
    for point3D in points3D:
        for track in point3D.track:
            images[track[0]-1].used_keypoints_idx.append((track[1], point3D.id, point3D.xyz))

def mde_depth_map(train_images_path, images: Image):
    repo = "isl-org/ZoeDepth"
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)

    with open(train_images_path, 'r') as f:
        for image in images:
            path = os.path.join(train_images_path, image.name)
            img = pil_image.open(path).convert('RGB')
            depth_numpy = zoe.infer_pil(img)       

if __name__ == '__main__':
    point3d_txt_path = 'D:\\PythonCode\\main_project\\fern_san\\3_views\\triangulated\\points3D.txt'
    img_txt_path = 'D:\\PythonCode\\main_project\\fern_san\\3_views\\triangulated\\images.txt'
    camera_txt_path = 'D:\\PythonCode\\main_project\\fern_san\\3_views\\triangulated\\cameras.txt'
    points_3d = read_point3d(point3d_txt_path)
    images = read_images(img_txt_path)
    set_used_keypoints(images, points_3d)
    camera_intrinsics = read_intrinsics_text(camera_txt_path)
    
    for image in images:
        image.camera_intrinsics = camera_intrinsics[image.camera_id]

    for image  in images:
        image.create_depth_map_pcd()
        print("hehe")