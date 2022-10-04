
import os, struct
import numpy as np
import zlib
import imageio
import cv2
import json
import subprocess

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}
RAW_SCAN_FOLDER = 'ScanNet/scannet_raw/scans/'
DEST_UNZIP_FOLDER = 'ScanNet/scans_processed'
SCENE_NAMES = sorted(os.listdir(RAW_SCAN_FOLDER))
TRAIN_SCENES = np.asarray(np.loadtxt("ScanNet/scannet_raw/scannetv2_train.txt", dtype='str'))
VAL_SCENES = np.asarray(np.loadtxt("ScanNet/scannet_raw/scannetv2_validation.txt", dtype='str'))
DEST_SPLIT_FOLDER = 'ScanNet/scans_split'

if not os.path.exists(DEST_UNZIP_FOLDER):
    os.makedirs(DEST_UNZIP_FOLDER)
if not os.path.exists(DEST_SPLIT_FOLDER):
    os.makedirs(DEST_SPLIT_FOLDER)
'''
This class is used to process the sensor data from ScanNet
INPUT:
- location of .sens file
'''
class SensorData:

  def __init__(self, filename):
    self.version = 4
    self.load(filename)


  def load(self, filename):
    print('start loading', filename)
    with open(filename, 'rb') as f:
      version = struct.unpack('I', f.read(4))[0]
      assert self.version == version
      strlen = struct.unpack('Q', f.read(8))[0]
      self.sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen)))
      self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
      self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
      self.color_width = struct.unpack('I', f.read(4))[0]
      self.color_height =  struct.unpack('I', f.read(4))[0]
      self.depth_width = struct.unpack('I', f.read(4))[0]
      self.depth_height =  struct.unpack('I', f.read(4))[0]
      self.depth_shift =  struct.unpack('f', f.read(4))[0]
      num_frames =  struct.unpack('Q', f.read(8))[0]
      self.frames = []
      for i in range(num_frames):
        frame = RGBDFrame()
        frame.load(f)
        self.frames.append(frame)


  def export_depth_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, ' depth frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
      depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
      if image_size is not None:
        depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)


  def export_color_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      color = self.frames[f].decompress_color(self.color_compression_type)
      if image_size is not None:
        color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f) + '.png'), color)


  def save_mat_to_file(self, matrix, filename):
    with open(filename, 'w') as f:
      for line in matrix:
        np.savetxt(f, line[np.newaxis], fmt='%f')


  def export_poses(self, output_path, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'camera poses to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, str(f) + '.txt'))


  def export_intrinsics(self, output_path):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting camera intrinsics to', output_path)
    self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
    self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
    self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
    self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))

class RGBDFrame():

  def load(self, file_handle):
    self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
    self.depth_data = b''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))


  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise ValueError("invalid type")


  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)


  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise ValueError("invalid type")


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


'''
This function is used to parse 2d images from ScanNet sensor data
'''
def parse_2d_images():
    for scene in SCENE_NAMES:
        sensor_data_file = os.path.join(RAW_SCAN_FOLDER, scene, scene+'.sens')
        output_scene_folder = os.path.join(DEST_UNZIP_FOLDER, scene)
        if os.path.exists(os.path.join(output_scene_folder, 'color')):
            continue
        if not os.path.exists(output_scene_folder):
            os.makedirs(output_scene_folder)
        sensor_data = SensorData(sensor_data_file)
        sensor_data.export_color_images(os.path.join(output_scene_folder, 'color'))

'''
This function is used to unzip raw 2d instance label (filterd version) in ScanNet
'''
def parse_seg_masks():
    for scene in SCENE_NAMES:
        source_zip = os.path.join(RAW_SCAN_FOLDER, scene, scene + '_2d-instance-filt.zip')
        output_scene_folder = os.path.join(DEST_UNZIP_FOLDER, scene)
        if os.path.exists(os.path.join(output_scene_folder, 'instance-filt')):
            continue
        if not os.path.exists(output_scene_folder):
            os.makedirs(output_scene_folder)
        subprocess.run(["unzip", source_zip, "-d", output_scene_folder])
        print('from', source_zip, 'to', output_scene_folder)

'''
This function is used to parse the offical train/val split downloaded from: https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark
The output are two json files consisting for train set and validation set separately
In each json file, keys are filenames for image, values are filenames for corresponding masks
'''
def parse_scene_split():
    train_split = {}
    val_split = {}
    for scene in SCENE_NAMES:
        if scene in TRAIN_SCENES:
            image_folder = os.path.join(DEST_UNZIP_FOLDER, scene, 'color')
            image_fname_list = os.listdir(image_folder)
            image_fname_list = sorted(image_fname_list, key=lambda x: int(x.split('.')[0]))
            for idx, image_fname in enumerate(image_fname_list):
                if idx % 20 != 0:
                    continue
                image_path = os.path.join(image_folder, image_fname)
                mask_path = os.path.join(DEST_UNZIP_FOLDER, scene, 'instance-filt', image_fname)
                train_split[image_path] = mask_path
        elif scene in VAL_SCENES:
            image_folder = os.path.join(DEST_UNZIP_FOLDER, scene, 'color')
            image_fname_list = os.listdir(image_folder)
            image_fname_list = sorted(image_fname_list, key=lambda x: int(x.split('.')[0]))
            for idx, image_fname in enumerate(image_fname_list):
                if idx % 20 != 0:
                    continue
                image_path = os.path.join(image_folder, image_fname)
                mask_path = os.path.join(DEST_UNZIP_FOLDER, scene, 'instance-filt', image_fname)
                val_split[image_path] = mask_path
        else:
            print('skip', scene)
    with open(os.path.join(DEST_SPLIT_FOLDER, 'train_split.json'), 'w') as f:
        json.dump(train_split, f, indent=2)
    with open(os.path.join(DEST_SPLIT_FOLDER, 'val_split.json'), 'w') as f:
        json.dump(val_split, f, indent=2)

if __name__ =='__main__':
   parse_2d_images()
   parse_seg_masks()
   parse_scene_split()
