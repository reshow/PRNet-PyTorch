import numpy as np
from data import UVmap2Mesh, getLandmark
from visualize import showImage, showMesh, show
import matplotlib.pyplot as plt
import pyrender, trimesh
from faceutil import mesh
import cv2

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1

IMAGE_WIDTH = 256
r = pyrender.OffscreenRenderer(IMAGE_WIDTH, IMAGE_WIDTH)
scene = pyrender.Scene()


def light_test(vertices, light_positions, light_intensities, triangles, colors, bg=None, h=256, w=256):
    lit_colors = mesh.light.add_light(vertices, triangles, colors, light_positions, light_intensities)
    # image_vertices = mesh.transform.to_image(vertices, h, w)
    rendering = mesh.render.render_colors(vertices, triangles, lit_colors, h, w, BG=bg)
    # rendering = np.minimum((np.maximum(rendering, 0)), 1)
    return rendering


def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1  # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:

        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], 255 - vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)


def renderLightBack(posmap, init_image=None):
    tex = np.ones((256, 256, 3)) / 2
    mesh = UVmap2Mesh(posmap, tex)
    vertices = mesh['vertices']
    triangles = mesh['triangles']
    colors = mesh['colors'] / np.max(mesh['colors'])
    showMesh(mesh)

    light_intensities = np.array([[1, 1, 1]])
    for i, p in enumerate(range(-200, 201, 60)):
        light_positions = np.array([[p, -100, 300]])
        image = light_test(vertices, light_positions, light_intensities, triangles, colors, bg=init_image)
        showImage(image)


def renderLight(posmap, init_image=None, is_render=True):
    tex = np.ones((256, 256, 3)) / 2
    mesh = UVmap2Mesh(posmap, tex, is_extra_triangle=False)
    vertices = mesh['vertices']
    triangles = mesh['triangles']
    colors = mesh['colors'] / np.max(mesh['colors'])
    file = 'tmp/light/test.obj'
    write_obj_with_colors(file, vertices, triangles, colors)

    obj = trimesh.load(file)
    # obj.visual.vertex_colors = np.random.uniform(size=obj.vertices.shape)
    obj.visual.face_colors = np.array([0.05, 0.1, 0.2])

    mesh = pyrender.Mesh.from_trimesh(obj, smooth=False)

    scene.add(mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[0, 3] = 128
    camera_pose[1, 3] = 128
    camera_pose[2, 3] = 300
    camera = pyrender.OrthographicCamera(xmag=128, ymag=128, zfar=1000)

    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=8.0)
    scene.add(light, pose=camera_pose)
    color, depth = r.render(scene)
    if is_render:
        plt.imshow(color)
        plt.show()

    if init_image is not None:
        sum_mask = np.mean(color, axis=-1)
        fuse_img = color.copy()
        fuse_img[sum_mask > 128] = init_image[sum_mask > 128]
        if is_render:
            plt.imshow(fuse_img)
            plt.show()
        scene.clear()
        return fuse_img

    scene.clear()
    return color


def plot_kpt(image, kpt, is_render=True, color_rate=0):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, (0 + color_rate, 0, 255 - color_rate), 2)
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (0 + color_rate, 0, 255 - color_rate), 1)
    if is_render:
        showImage(image)
    return image


def demoKpt(posmap, image, is_render=True):
    kpt = getLandmark(posmap)
    ploted = plot_kpt(image, kpt, is_render=is_render)
    return ploted


def compareKpt(posmap, gtposmap, image, is_render=True):
    kpt1 = getLandmark(posmap)
    kpt2 = getLandmark(gtposmap)
    ploted = plot_kpt(image, kpt1, is_render=is_render)
    ploted = plot_kpt(ploted, kpt2, is_render=is_render, color_rate=int(255))
    return ploted


def demoAll(posmap, image, is_render=True):
    return renderLight(posmap, image.copy(), is_render=is_render), demoKpt(posmap, image.copy(), is_render=is_render)


if __name__ == '__main__':
    pass
    pos = np.load('data/images/AFLW2000-full/image00004/image00004_cropped_uv_posmap.npy')
    img = np.load('data/images/AFLW2000-full/image00004/image00004_cropped.npy')
    renderLight(pos, img)
