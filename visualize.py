import numpy as np
import scipy.io as sio
from skimage import io
from faceutil import mesh
# from data import bfm, modelParam2Mesh, UVMap2Mesh
import matplotlib.pyplot as plt
from skimage import io, transform
from data import UVmap2Mesh, uv_kpt, bfm2Mesh, getLandmark, mesh2UVmap, bfm


def showLandmark(image, kpt):
    kpt = np.round(kpt).astype(np.int)
    image[kpt[:, 1], kpt[:, 0]] = np.array([1, 0, 0])
    image[kpt[:, 1] + 1, kpt[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt[:, 1] - 1, kpt[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt[:, 1] - 1, kpt[:, 0] - 1] = np.array([1, 0, 0])
    image[kpt[:, 1] + 1, kpt[:, 0] - 1] = np.array([1, 0, 0])
    plt.imshow(image)
    plt.show()


def showLandmark2(image, kpt1, kpt2):
    kpt1 = np.round(kpt1).astype(np.int)
    kpt2 = np.round(kpt2).astype(np.int)
    image[kpt1[:, 1], kpt1[:, 0]] = np.array([1, 0, 0])
    image[kpt1[:, 1] + 1, kpt1[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt1[:, 1] - 1, kpt1[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt1[:, 1] - 1, kpt1[:, 0] - 1] = np.array([1, 0, 0])
    image[kpt1[:, 1] + 1, kpt1[:, 0] - 1] = np.array([1, 0, 0])

    image[kpt2[:, 1], kpt2[:, 0]] = np.array([0, 1, 0])
    image[kpt2[:, 1] + 1, kpt2[:, 0]] = np.array([0, 1, 0])
    image[kpt2[:, 1] - 1, kpt2[:, 0]] = np.array([0, 1, 0])
    image[kpt2[:, 1], kpt2[:, 0] + 1] = np.array([0, 1, 0])
    image[kpt2[:, 1], kpt2[:, 0] - 1] = np.array([0, 1, 0])

    plt.imshow(image)
    plt.show()


def showGTLandmark(image_path):
    image = io.imread(image_path) / 255.0
    bfm_info = sio.loadmat(image_path.replace('jpg', 'mat'))
    if 'pt3d_68' in bfm_info.keys():
        kpt = bfm_info['pt3d_68'].T
    else:
        kpt = bfm_info['pt2d'].T
    showLandmark(image, kpt)

    mesh_info = bfm2Mesh(bfm_info, image.shape)

    kpt2 = mesh_info['vertices'][bfm.kpt_ind]
    showLandmark2(image, kpt, kpt2)
    return kpt, kpt2


def showImage(image, is_path=False):
    if is_path:
        img = io.imread(image) / 255.
        io.imshow(img)
        plt.show()
    else:
        io.imshow(image)
        plt.show()


def showMesh(mesh_info, init_img=None):
    height = np.ceil(np.max(mesh_info['vertices'][:, 1])).astype(int)
    width = np.ceil(np.max(mesh_info['vertices'][:, 0])).astype(int)
    channel = 3
    if init_img is not None:
        [height, width, channel] = init_img.shape
    mesh_image = mesh.render.render_colors(mesh_info['vertices'], mesh_info['triangles'], mesh_info['colors'],
                                           height, width, channel)
    if init_img is None:
        io.imshow(mesh_image)
        plt.show()
    else:
        plt.subplot(1, 3, 1)
        plt.imshow(mesh_image)

        plt.subplot(1, 3, 3)
        plt.imshow(init_img)

        verify_img = mesh.render.render_colors(mesh_info['vertices'], mesh_info['triangles'], mesh_info['colors'],
                                               height, width, channel, BG=init_img)
        plt.subplot(1, 3, 2)
        plt.imshow(verify_img)

        plt.show()


def show(ipt, is_file=False, mode='image'):
    if mode == 'image':
        if is_file:
            # ipt is a path
            image = io.imread(ipt) / 255.
        else:
            image = ipt
        io.imshow(image)
        plt.show()
    elif mode == 'uvmap':
        # ipt should be [posmap texmap] or [posmap texmap image]
        assert (len(ipt) > 1)
        init_image = None
        if is_file:
            uv_position_map = np.load(ipt[0])
            uv_texture_map = io.imread(ipt[1]) / 255.
            if len(ipt) > 2:
                init_image = io.imread(ipt[2]) / 255.
        else:
            uv_position_map = ipt[0]
            uv_texture_map = ipt[1]
            if len(ipt) > 2:
                init_image = ipt[2]
        mesh_info = UVmap2Mesh(uv_position_map=uv_position_map, uv_texture_map=uv_texture_map)
        showMesh(mesh_info, init_image)
    elif mode == 'mesh':
        if is_file:
            if len(ipt) == 2:
                mesh_info = sio.loadmat(ipt[0])
                init_image = io.imread(ipt[1]) / 255.
            else:
                mesh_info = sio.loadmat(ipt)
                init_image = None
        else:
            if len(ipt == 2):
                mesh_info = ipt[0]
                init_image = ipt[1]
            else:
                mesh_info = ipt
                init_image = None
        showMesh(mesh_info, init_image)


if __name__ == "__main__":
    pass
    # showUVMap('data/images/AFLW2000-out/image00002/image00002_uv_posmap.npy', None,
    #           # 'data/images/AFLW2000-output/image00002/image00002_uv_texture_map.jpg',
    #           'data/images/AFLW2000-out/image00002/image00002_init.jpg', True)
    # show(['data/images/AFLW2000-crop-offset/image00002/image00002_cropped_uv_posmap.npy',
    #       'data/images/AFLW2000-crop/image00002/image00002_uv_texture_map.jpg',
    #       'data/images/AFLW2000-crop-offset/image00002/image00002_cropped.jpg'], is_file=True, mode='uvmap')
