import numpy as np
import open3d as o3d

def read_obj(path):
    """
    read verts and faces from obj file. This func will convert quad mesh to triangle mesh
    """
    with open(path) as f:
        lines = f.read().splitlines()
    verts = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            verts.append(np.array([float(k) for k in line.split(' ')[1:]]))
        elif line.startswith('f '):

            # breakpoint()
            if '//' in line:
                try:
                    onef = np.array([int(k.split('//')[0]) for k in line.split(' ')[1:]])
                    faces.append(onef)
                except ValueError:
                    continue
            else:
                try:
                    onef = np.array([int(k) for k in line.split(' ')[1:]])
                except ValueError:
                    continue
                if len(onef) == 4:
                    faces.append(onef[[0, 1, 2]])
                    faces.append(onef[[0, 2, 3]])
                elif len(onef) > 4:
                    pass
                else:
                    faces.append(onef)

    # print(len(faces))
    if len(faces) == 0:
        return np.stack(verts), None
    else:
        return np.stack(verts), np.stack(faces)-1

class MeshO3d(object):
    def __init__(self, v=None, f=None, filename=None):
        self.m = o3d.geometry.TriangleMesh()
        if v is not None:
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
        elif filename is not None:
            v, f = read_obj(filename)
            # breakpoint()
            self.m = o3d.geometry.TriangleMesh()
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))

    @property
    def v(self):
        return np.asarray(self.m.vertices)

    @v.setter
    def v(self, value):
        self.m.vertices = o3d.utility.Vector3dVector(value.astype(np.float32))

    @property
    def f(self):
        return np.asarray(self.m.triangles)

    @f.setter
    def f(self, value):
        self.m.triangles = o3d.utility.Vector3iVector(value.astype(np.int32))

    def write_obj(self, fpath):
        if not fpath.endswith('.obj'):
            fpath = fpath + '.obj'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=True)

    def write_ply(self, fpath):
        if not fpath.endswith('.ply'):
            fpath = fpath + '.ply'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=False, compressed=True)