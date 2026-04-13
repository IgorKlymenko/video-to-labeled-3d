#!/usr/bin/env python3
## View a .ply / .obj mesh with Open3D. usage: python show_mesh.py <path>
import os
import sys

import open3d as o3d


def main():
    if len(sys.argv) < 2:
        print("usage: python show_mesh.py <mesh.ply|mesh.obj>")
        print("  Example: python show_mesh.py ../out_apple/mesh_class_colored.ply")
        sys.exit(1)
    path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(path):
        print(f"file not found: {path}")
        print()
        print("--yolo + --mesh => UV texture from photo + class overlay.")
        print("see class_legend.png for class names.")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        geom = o3d.io.read_triangle_mesh(path)
        if len(geom.vertices) == 0:
            geom = o3d.io.read_point_cloud(path)
    else:
        geom = o3d.io.read_triangle_mesh(path)

    if isinstance(geom, o3d.geometry.TriangleMesh):
        geom.compute_vertex_normals()

    o3d.visualization.draw_geometries([geom])


if __name__ == "__main__":
    main()
