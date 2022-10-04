import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont
import cv2
import torchvision.transforms as transforms

from utils.misc import to_np

bbox_color_list=np.zeros([21, 3]).astype(np.uint8)
bbox_color_list[0,:] = np.array([ 240,  10,  20])
bbox_color_list[5,:] = np.array([ 0,  50,  80])
bbox_color_list[6,:] = np.array([244, 35,232])
bbox_color_list[15,:] = np.array([ 70, 70, 70])
bbox_color_list[9,:] = np.array([ 102,102,156])
bbox_color_list[4,:] = np.array([ 190,153,153])

bbox_color_list[1,:] = np.array([ 250,170, 30])
bbox_color_list[7,:] = np.array([ 220,220,  0])
bbox_color_list[8,:] = np.array([ 107,142, 35])
bbox_color_list[3,:] = np.array([ 152,251,152])
bbox_color_list[10,:] = np.array([ 70,130,180])

bbox_color_list[11,:] = np.array([ 220, 20, 60])
bbox_color_list[12,:] = np.array([ 119, 11, 32])
bbox_color_list[13,:] = np.array([ 0,  0,142])
bbox_color_list[14,:] = np.array([  0,  0, 70])
bbox_color_list[2,:] = np.array([  0, 30,240])

bbox_color_list[16,:] = np.array([  0, 80,100])
bbox_color_list[17,:] = np.array([  0,  0,230])
bbox_color_list[18,:] = np.array([ 255,  0,  0])
bbox_color_list[19,:] = np.array([ 80,  0,  80])
bbox_color_list[20,:] = np.array([ 80,  120,  200])
bbox_color_list = bbox_color_list/255.0

class SpatialTransformer:
    def __init__(self, input_shape, output_shape):
        """
        :param input_shape: (H, W)
        :param output_shape: (H, W)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _transform(self, x, z_where, inverse):
        """
        :param x: (B, 1, Hin, Win)
        :param z_where: [s, x, y]
        :param inverse: inverse z_where
        :return: y of output_size
        """
        if inverse:
            z_where = invert_z_where(z_where)
            out_shp = self.input_shape
        else:
            out_shp = self.output_shape

        out = spatial_transformer(x, z_where, out_shp)
        return out

    def forward(self, x, z_where):
        return self._transform(x, z_where, inverse=False)

    def inverse(self, x, z_where):
        return self._transform(x, z_where, inverse=True)


def spatial_transformer(x, z_where, out_shape):
    """
    Resamples x on a grid of shape out_shape based on an affine transform
    parameterized by z_where.
    The output image has shape out_shape.

    :param x:
    :param z_where:
    :param out_shape:
    :return:
    """
    batch_sz = x.size(0)
    theta = expand_z_where(z_where)
    grid_shape = torch.Size((batch_sz, 1) + out_shape)
    grid = F.affine_grid(theta, grid_shape, align_corners=False)
    out = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
    return out

def expand_z_where(z_where):
    """
    :param z_where: batch. [s, x, y]
    :return: [[s, 0, x], [0, s, y]]
    """
    bs = z_where.size(0)
    dev = z_where.device

    # [s, x, y] -> [s, 0, x, 0, s, y]
    z_where = torch.cat((torch.zeros(bs, 1, device=dev), z_where), dim=1)
    expansion_indices = torch.tensor([1, 0, 2, 0, 1, 3], device=dev)
    matrix = torch.index_select(z_where, dim=1, index=expansion_indices)
    matrix = matrix.view(bs, 2, 3)

    return matrix

def invert_z_where(z_where):
    z_where_inv = torch.zeros_like(z_where)
    scale = z_where[:, 0:1]   # (batch, 1)
    z_where_inv[:, 1:3] = -z_where[:, 1:3] / scale   # (batch, 2)
    z_where_inv[:, 0:1] = 1 / scale    # (batch, 1)
    return z_where_inv


def batch_add_bounding_boxes(imgs, z_wheres, n_obj, color=None, n_img=None, gt_labels=None):
    """

    :param imgs: 4d tensor of numpy array, channel dim either 1 or 3
    :param z_wheres: tensor or numpy of shape (n_imgs, max_n_objects, 3)
    :param n_obj:
    :param color:
    :param n_img:
    :param gt_labels: shape [B, max_step, 1]
    :return:
    """

    # Check arguments
    assert len(imgs.shape) == 4
    assert imgs.shape[1] in [1, 3]
    assert len(z_wheres.shape) == 3
    assert z_wheres.shape[0] == imgs.shape[0]
    assert z_wheres.shape[2] == 3

    target_shape = list(imgs.shape)
    target_shape[1] = 3

    if n_img is None:
        n_img = len(imgs)
    if color is None:
        color = np.array([1., 0., 0.])
    if gt_labels is not None:
        out = torch.stack([
            add_bounding_boxes(imgs[j], z_wheres[j], color, n_obj[j], gt_labels[j])
            for j in range(n_img)
        ])
    else:
        out = torch.stack([
            add_bounding_boxes(imgs[j], z_wheres[j], color, n_obj[j])
            for j in range(n_img)
        ])

    out_shape = tuple(out.shape)
    target_shape = tuple(target_shape)
    assert out_shape == target_shape, "{}, {}".format(out_shape, target_shape)
    return out


def add_bounding_boxes(img, z_wheres, color, n_obj, gt_labels=None):
    """
    Adds bounding boxes to the n_obj objects in img, according to z_wheres.
    The output is never on cuda.

    :param img: image in 3d or 4d shape, either Tensor or numpy. If 4d, the
                first dimension must be 1. The channel dimension must be
                either 1 or 3.
    :param z_wheres: tensor or numpy of shape (1, max_n_objects, 3) or
                (max_n_objects, 3)
    :param color: color of all bounding boxes (RGB)
    :param n_obj: number of objects in the scene. This controls the number of
                bounding boxes to be drawn, and cannot be greater than the
                max number of objects supported by z_where (dim=1). Has to be
                a scalar or a single-element Tensor/array.
    :param gt_digit: gt label for inferred objects
    :return: image with required bounding boxes, with same type and dimension
                as the original image input, except 3 color channels.
    """

    try:
        n_obj = n_obj.item()
    except AttributeError:
        pass
    n_obj = int(round(n_obj))
    # assert n_obj <= z_wheres.shape[1]

    try:
        img = img.cpu()
    except AttributeError:
        pass

    if len(img.shape) == 3:
        color_dim = 0
    else:
        color_dim = 1

    if len(z_wheres.shape) == 3:
        assert z_wheres.shape[0] == 1
        z_wheres = z_wheres[0]

    target_shape = list(img.shape)
    target_shape[color_dim] = 3

    if gt_labels is not None:
        for i in range(n_obj):
            img = add_bounding_box(img, z_wheres[i:i+1], color=bbox_color_list[i], gt_label=gt_labels[i])
    else:
        for i in range(n_obj):
            img = add_bounding_box(img, z_wheres[i:i+1], color=bbox_color_list[i])
    if img.shape[color_dim] == 1:  # this might happen if n_obj==0
        reps = [3, 1, 1]
        if color_dim == 1:
            reps = [1] + reps
        reps = tuple(reps)
        if isinstance(img, torch.Tensor):
            img = img.repeat(*reps)
        else:
            img = np.tile(img, reps)

    target_shape = tuple(target_shape)
    img_shape = tuple(img.shape)
    assert img_shape == target_shape, "{}, {}".format(img_shape, target_shape)
    return img

def _bounding_box(z_where, x_size, rounded=True, margin=1):
        z_where = to_np(z_where).flatten()
        assert z_where.shape[0] == z_where.size == 3
        s, x, y = tuple(z_where)
        w = x_size / s
        h = x_size / s
        xtrans = -x / s * x_size / 2
        ytrans = -y / s * x_size / 2
        x1 = (x_size - w) / 2 + xtrans - margin
        y1 = (x_size - h) / 2 + ytrans - margin
        x2 = x1 + w + 2 * margin
        y2 = y1 + h + 2 * margin
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        coords = (x1, x2, y1, y2)
        if rounded:
            coords = (int(round(t)) for t in coords)
        return coords


def add_bounding_box(img, z_where, color, gt_label=None):
    """
    Adds a bounding box to img with parameters z_where and the given color.
    Makes a copy of the input image, which is left unaltered. The output is
    never on cuda.

    :param img: image in 3d or 4d shape, either Tensor or numpy. If 4d, the
                first dimension must be 1. The channel dimension must be
                either 1 or 3.
    :param z_where: tensor or numpy with 3 elements, and shape (1, ..., 1, 3)
    :param color:
    :return: image with required bounding box in the specified color, with same
                type and dimension as the original image input, except 3 color
                channels.
    """
    # def _bounding_box(z_where, x_size, rounded=True, margin=1):
    #     print('z where', z_where)
    #     z_where = to_np(z_where).flatten()
    #     assert z_where.shape[0] == z_where.size == 3
    #     s, x, y = tuple(z_where)
    #     w = x_size / s
    #     h = x_size / s
    #     xtrans = -x / s * x_size / 2
    #     ytrans = -y / s * x_size / 2
    #     x1 = (x_size - w) / 2 + xtrans - margin
    #     y1 = (x_size - h) / 2 + ytrans - margin
    #     x2 = x1 + w + 2 * margin
    #     y2 = y1 + h + 2 * margin
    #     x1, x2 = sorted((x1, x2))
    #     y1, y2 = sorted((y1, y2))
    #     coords = (x1, x2, y1, y2)
    #     if rounded:
    #         coords = (int(round(t)) for t in coords)
    #     return coords

    target_shape = list(img.shape)
    collapse_first = False
    torch_tensor = isinstance(img, torch.Tensor)
    img = to_np(img).copy()
    if len(img.shape) == 3:
        collapse_first = True
        img = np.expand_dims(img, 0)
        target_shape[0] = 3
    else:
        target_shape[1] = 3
    assert len(img.shape) == 4 and img.shape[0] == 1
    if img.shape[1] == 1:
        img = np.tile(img, (1, 3, 1, 1))
    assert img.shape[1] == 3
    color = color[:, None]

    x1, x2, y1, y2 = _bounding_box(z_where, img.shape[2])
    x_max = y_max = img.shape[2] - 1

    if 0 <= y1 <= y_max:
        img[0, :, y1, max(x1, 0):min(x2, x_max)] = color
    if 0 <= y2 - 1 <= y_max:
        img[0, :, y2 - 1, max(x1, 0):min(x2, x_max)] = color
    if 0 <= x1 <= x_max:
        img[0, :, max(y1, 0):min(y2, y_max), x1] = color
    if 0 <= x2 - 1 <= x_max:
        img[0, :, max(y1, 0):min(y2, y_max), x2 - 1] = color

    # if gt_label is not None:
    #     cv2.imwrite('img1.png', img[0])
    #     img[0] = cv2.putText(img[0], str(gt_label.item()), (0,0), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
    #     cv2.imwrite('img2.png', img[0])
    #     input()

    if collapse_first:
        img = img[0]
    if torch_tensor:
        img = torch.from_numpy(img)
    
    if gt_label is not None:
        to_pil = transforms.ToPILImage()
        img = to_pil(img)
        draw = ImageDraw.Draw(img)
        draw.text((x1, y1), str(int(gt_label.item())), (255,0,0))
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)

    target_shape = tuple(target_shape)
    img_shape = tuple(img.shape)
    assert img_shape == target_shape, "{}, {}".format(img_shape, target_shape)
    return img

def _test(obj_size, canvas_size, color_ch):

    # Object to image. Meaningful scenario:
    #   scale > 1, x and y in [-scale, +scale]
    # Perfect copy (no interpolation) when scale == canvas_size / obj_size
    obj = (torch.rand(1, color_ch, obj_size, obj_size) < 0.8).float()
    z_where = torch.tensor([[6., 2., 4.]])
    spatial_transf = SpatialTransformer(
        (obj_size, obj_size), (canvas_size, canvas_size))
    out = spatial_transf.forward(obj, z_where)

    plt.figure()
    plt.imshow(out[0].permute(1, 2, 0).squeeze(), vmin=0., vmax=1.)
    plt.show()

    # Image to object.
    # Here we retrieve the same object we initially drew on the canvas.
    img = out
    out = spatial_transf.inverse(img, z_where)

    plt.figure()
    plt.imshow(out[0].permute(1, 2, 0).squeeze(), vmin=0., vmax=1.)
    plt.show()

    # show bounding box
    img_np = to_np(img)
    color = np.array([1., 0., 0.])
    img_np = add_bounding_box(img_np, z_where, color)
    plt.figure()
    plt.imshow(img_np[0].transpose(1, 2, 0), vmin=0., vmax=1.)
    plt.show()

    # test bounding box methods
    add_bounding_box(img[0], z_where, color)  # 3d tensor
    add_bounding_box(img_np[0], z_where, color)  # 3d numpy
    add_bounding_box(img, z_where, color)  # 4d tensor
    add_bounding_box(img_np, z_where, color)  # 4d numpy
    z_wheres = z_where.repeat(1, 4, 1)
    z_wheres += torch.randn_like(z_wheres) * 2
    add_bounding_boxes(img[0], z_wheres, color, 4)  # 3d tensor
    add_bounding_boxes(img_np[0], z_wheres, color, 4)  # 3d numpy
    add_bounding_boxes(img, z_wheres, color, 4)  # 4d tensor
    img_np = add_bounding_boxes(img_np, z_wheres, color, 4)  # 4d numpy
    plt.figure()
    plt.imshow(img_np[0].transpose(1, 2, 0), vmin=0., vmax=1.)
    plt.show()

    # case nobj = 0 missing

def batch_add_step_bounding_boxes(imgs, z_wheres, n_obj, color=None, n_img=None, step=None):
    """

    :param imgs: 4d tensor of numpy array, channel dim either 1 or 3
    :param z_wheres: tensor or numpy of shape (n_imgs, max_n_objects, 3)
    :param n_obj:
    :param color:
    :param n_img:
    :return:
    """

    # Check arguments
    assert len(imgs.shape) == 4
    assert imgs.shape[1] in [1, 3]
    assert len(z_wheres.shape) == 3
    assert z_wheres.shape[0] == imgs.shape[0]
    assert z_wheres.shape[2] == 3

    target_shape = list(imgs.shape)
    target_shape[1] = 3

    if n_img is None:
        n_img = len(imgs)
    if color is None:
        color = np.array([1., 0., 0.])
    out = torch.stack([
        add_step_bounding_box(imgs[j], z_wheres[j], color, n_obj[j], step=step)
        for j in range(n_img)
    ])

    out_shape = tuple(out.shape)
    target_shape = tuple(target_shape)
    assert out_shape == target_shape, "{}, {}".format(out_shape, target_shape)
    return out


def add_step_bounding_box(img, z_wheres, color, n_obj, step):
    """
    Adds bounding boxes to the n_obj objects in img, according to z_wheres.
    The output is never on cuda.

    :param img: image in 3d or 4d shape, either Tensor or numpy. If 4d, the
                first dimension must be 1. The channel dimension must be
                either 1 or 3.
    :param z_wheres: tensor or numpy of shape (1, max_n_objects, 3) or
                (max_n_objects, 3)
    :param color: color of all bounding boxes (RGB)
    :param n_obj: number of objects in the scene. This controls the number of
                bounding boxes to be drawn, and cannot be greater than the
                max number of objects supported by z_where (dim=1). Has to be
                a scalar or a single-element Tensor/array.
    :return: image with required bounding boxes, with same type and dimension
                as the original image input, except 3 color channels.
    """

    try:
        n_obj = n_obj.item()
    except AttributeError:
        pass
    n_obj = int(round(n_obj))
    # print(n_obj, z_wheres.shape)
    # assert n_obj <= z_wheres.shape[1]

    try:
        img = img.cpu()
    except AttributeError:
        pass

    if len(img.shape) == 3:
        color_dim = 0
    else:
        color_dim = 1

    if len(z_wheres.shape) == 3:
        assert z_wheres.shape[0] == 1
        z_wheres = z_wheres[0]

    target_shape = list(img.shape)
    target_shape[color_dim] = 3

    # for i in range(n_obj):
    if step < n_obj:
        img = add_bounding_box(img, z_wheres[step:step+1], color=bbox_color_list[step])
    if img.shape[color_dim] == 1:  # this might happen if n_obj==0
        reps = [3, 1, 1]
        if color_dim == 1:
            reps = [1] + reps
        reps = tuple(reps)
        if isinstance(img, torch.Tensor):
            img = img.repeat(*reps)
        else:
            img = np.tile(img, reps)

    target_shape = tuple(target_shape)
    img_shape = tuple(img.shape)
    assert img_shape == target_shape, "{}, {}".format(img_shape, target_shape)
    return img

def int_clamp(x, low, upper):
    if x < low:
        return low
    if x > upper:
        return upper
    return x

def cut_out_objects(img, z_wheres):
    '''
    cut object (based on bbox derived from z_where) out from image and output the remaining background
    img: the original whole image ready for cutting object out [3, image_size, image_size]
    z_wheres: [max_steps, 3]
    '''
    max_steps = z_wheres.shape[0]
    black = torch.FloatTensor(0).to(img.device)
    x_max = y_max = img.shape[2] - 1
    for step in range (0, max_steps):
        x1, x2, y1, y2 = _bounding_box(z_wheres[step], img.shape[2])
        channel = img.shape[0]
        x1 = int_clamp(x1, 0, x_max)
        x2 = int_clamp(x2-1, 0, x_max)
        y1 = int_clamp(y1, 0, y_max)
        y2 = int_clamp(y2-1, 0, y_max)
        mask = torch.ones(channel, img.shape[1], img.shape[2], device=img.device)
        mask[:, y1:y2, x1:x2] = torch.zeros(channel, (y2-y1), (x2-x1))
        img = img * mask
        # if 0 <= y1 <= y_max:
        #     img[:, y1, max(x1, 0):min(x2, x_max)] = black
        # if 0 <= y2 - 1 <= y_max:
        #     img[:, y2 - 1, max(x1, 0):min(x2, x_max)] = black
        # if 0 <= x1 <= x_max:
        #     img[:, max(y1, 0):min(y2, y_max), x1] = black
        # if 0 <= x2 - 1 <= x_max:
        #     img[:, max(y1, 0):min(y2, y_max), x2 - 1] = black
    return img

def batch_cut_out_objects(imgs, batch_z_wheres):
    bs = imgs.shape[0]
    out = torch.stack([
        cut_out_objects(imgs[j], batch_z_wheres[j])
        for j in range(bs)
    ])
    return out

