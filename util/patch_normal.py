import os
import shutil
import numpy as np

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

    def __init__(self, id: int, points2d: list[(float,float, int)], name: str) -> None:
        self.id = id
        self.points2d = points2d
        self.name = name
        self.used_keypoints_idx: list[(int,int, np.array[float,float,float])] = [] # (point2d_idx, point3d_id, xyz in 3d)
        self.depth_map_pcd: np.array = None

    def create_depth_map_pcd(self):
        x: int = None
        y: int = None
        self.depth_map_pcd = np.zeros(self.img_size)
        for elem in self.used_keypoints_idx:
            x = int(self.points2d[elem[0]][0])
            y = int(self.points2d[elem[0]][1])
            z_3D = elem[2][2]
            curr_z = self.depth_map_pcd[x, y]
            if curr_z < z_3D:
                self.depth_map_pcd[x, y] = z_3D
                print("Changed", end=" ")

def read_point3d(path):
    points: list[Point3D] = []
    with open(path, 'r') as f:
        for _ in range(3):
            f.readline()
        
        while True:
            line = f.readline()
            if not line:
                break
            tokens = line.split()
            point3d = Point3D()
            point3d.id = (int(tokens[0]))
            point3d.xyz = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
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
            #skip until 9
            name = tokens[9]
            tokens.clear()

            #read points2d
            line = f.readline()
            tokens = line.split()
            points2d = []
            for i in range(0, len(tokens), 3):
                x = float(tokens[i])
                y = float(tokens[i+1])
                point3d_id = int(tokens[i+2])
                points2d.append((x, y, point3d_id))
            images.append(Image(image_id, points2d, name))
    images.sort(key=lambda x: x.id)
    return images

def set_used_keypoints(images, points3D):
    for point3D in points3D:
        for track in point3D.track:
            images[track[0]-1].used_keypoints_idx.append((track[1], point3D.id, point3D.xyz))

if __name__ == '__main__':
    point3d_txt_path = 'D:\\PythonCode\\main_project\\fern_san\\3_views\\triangulated\\points3D.txt'
    img_txt_path = 'D:\\PythonCode\\main_project\\fern_san\\3_views\\triangulated\\images.txt'
    points_3d = read_point3d(point3d_txt_path)
    images = read_images(img_txt_path)
    set_used_keypoints(images, points_3d)

    for image  in images:
        image.create_depth_map_pcd()