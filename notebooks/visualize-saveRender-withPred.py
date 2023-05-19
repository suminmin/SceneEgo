import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pickle
import open3d
from copy import copy
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
# sys.path.append("../")
sys.path.append( os.path.dirname(sys.modules['__main__'].__file__) + "/..")
from utils import cfg
from utils.skeleton import Skeleton
from utils.depth2pointcloud import Depth2PointCloud

from dataset.test_dataset import TestDataset 


# CAMERA_CONFIG = '../utils/fisheye/fisheye.calibration_05_08.json'
CAMERA_CONFIG = 'utils/fisheye/fisheye.calibration_05_08.json'


def visualize(img_path, depth_path, pose_path):

    with open(pose_path, 'rb') as f:
        pose = pickle.load(f)

    visualize_data(img_path, depth_path, pose)


def visualize_data(img_path, depth_path, pose):
    scene, pose_mesh = get_visualize_data(img_path, depth_path, pose)
    open3d.visualization.draw_geometries([scene, pose_mesh])    

    
def get_visualize_data(img_path, depth_path, pose):
    skeleton = Skeleton(calibration_path=CAMERA_CONFIG)
    pose_mesh = skeleton.joints_2_mesh(pose)

    get_point_cloud = Depth2PointCloud(visualization=False,
                                       camera_model=CAMERA_CONFIG)

    scene = get_point_cloud.get_point_cloud_single_image(depth_path, img_path, output_path=None)
    return scene, pose_mesh


def blend_pcd_color(colors1, colors2, alpha):
    np_colors1 = np.asarray(colors1)
    np_colors2 = np.asarray(colors2)
    return open3d.utility.Vector3dVector( alpha*np_colors1 + (1.0-alpha)*np_colors2 )


def render_sequence(test_dataset_dir, SEQ_NAME):

    def get_dataset(seq_name):
        # config_path = '../experiments/sceneego/test/sceneego-mytest-jupyter.yaml'
        config_path = 'experiments/sceneego/test/sceneego-mytest.yaml'

        config = cfg.load_config(config_path)
        testDataset = TestDataset(config=config, root_dir=test_dataset_dir, seq_name=seq_name)
        
        image_path_list, gt_pose_list, depth_path_list = testDataset.get_gt_data(root_dir=test_dataset_dir, seq_name=seq_name)
        return testDataset, image_path_list, gt_pose_list, depth_path_list
    testDataset, image_path_list, gt_pose_list, depth_path_list = get_dataset(SEQ_NAME)


    def get_init_view():
        vis = open3d.visualization.Visualizer()
        vis.create_window(
            window_name="Hoge",  # ウインドウ名
            width=800,           # 幅
            height=600,          # 高さ
            left=50,             # 表示位置(左)
            top=50               # 表示位置(上)
        )
        ctr = vis.get_view_control()

        render_pose = open3d.io.read_pinhole_camera_trajectory("./notebooks/record_camera_trajectory.json")
        render_pose = render_pose.parameters[0]
        return vis, ctr, render_pose
    vis, ctr, render_pose = get_init_view()

    def get_pred_pose_data(estimated_depth_name, SEQ_NAME):
        save_pred_dir = f"./notebooks/test_inference_results/{SEQ_NAME}"
        if estimated_depth_name:
            save_result_filename = f'{save_pred_dir}/no_body_{SEQ_NAME}-{estimated_depth_name}.pkl'
        else:
            save_result_filename = f'{save_pred_dir}/no_body_{SEQ_NAME}-withGTdepth.pkl'

        with open(save_result_filename, 'rb') as f:
            predicted_joint_list = pickle.load(f)
        return np.asarray(predicted_joint_list)
    
    estimated_depth_name = None
    # estimated_depth_name = "matterport_green"

    predicted_joint_array = get_pred_pose_data(estimated_depth_name, SEQ_NAME)

    # save_render_img_dir = f"./notebooks/render_img/{SEQ_NAME}"
    save_render_img_dir = f"./notebooks/render_img_predWithGTdepth/{SEQ_NAME}"
    os.makedirs(save_render_img_dir, exist_ok=True)

    skeleton = Skeleton(calibration_path=CAMERA_CONFIG)
    get_point_cloud = Depth2PointCloud(visualization=False,
                                       camera_model=CAMERA_CONFIG)

    vis_scene_pcd_alpha = 0.5

    # skip = 1
    skip = 3
    for i,(img_path, gt_pose, depth_path, pred_pose) in enumerate( tqdm( zip(image_path_list[::skip], gt_pose_list[::skip], depth_path_list[::skip], predicted_joint_array[::skip]), total=len(image_path_list[::skip]) ) ):
        i_frame = i*skip

        scene = get_point_cloud.get_point_cloud_single_image(depth_path, img_path, output_path=None)
        gt_pose_mesh = skeleton.joints_2_mesh(gt_pose, joint_color=(0.7, 0.1, 0.1), bone_color=(0.9, 0.1, 0.1))
        pred_pose_mesh = skeleton.joints_2_mesh(pred_pose, joint_color=(0.1, 0.1, 0.7), bone_color=(0.1, 0.1, 0.9))

        if i == 0:
            render_scene = copy(scene)
            render_gt_pose_mesh = copy(gt_pose_mesh)
            render_pred_pose_mesh = copy(pred_pose_mesh)
            render_scene.colors = blend_pcd_color(render_scene.colors, np.ones( (len(render_scene.colors), 3) ), vis_scene_pcd_alpha)
            vis.add_geometry(render_scene)
            vis.add_geometry(render_gt_pose_mesh)
            vis.add_geometry(render_pred_pose_mesh)
            ctr.convert_from_pinhole_camera_parameters( render_pose, allow_arbitrary=True )
        else:
            render_scene.colors = scene.colors
            render_scene.points = scene.points
            render_gt_pose_mesh.vertices = gt_pose_mesh.vertices
            render_pred_pose_mesh.vertices = pred_pose_mesh.vertices
            render_scene.colors = blend_pcd_color(render_scene.colors, np.ones( (len(render_scene.colors), 3) ), vis_scene_pcd_alpha)
            vis.update_geometry(render_scene)
            vis.update_geometry(render_gt_pose_mesh)
            vis.update_geometry(render_pred_pose_mesh)
        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(False)
        plt.imsave(f"{save_render_img_dir}/scene3d_{i_frame:06d}.png", np.asarray(image), dpi = 1)

    vis.destroy_window()


# SEQ_NAME = "diogo1"
# render_sequence(SEQ_NAME)


sys.path.append("C:/workspace/sumiRepo/_lib_python")
import dirlist as DL


# test_dataset_dir = "E:/_dataset/SceneEgo/SceneEgo_test/SceneEgo_test"
test_dataset_dir = "./data/SceneEgo_test/SceneEgo_test"

dirList = DL.get_dirlist(test_dataset_dir)
for dirname in dirList:
    print(dirname)
    render_sequence(test_dataset_dir, dirname)
