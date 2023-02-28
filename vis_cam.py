import numpy as np
import cv2
import torch
import os
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from datetime import datetime
from util.camera_pose_visualizer import CameraPoseVisualizer

now = datetime.now()
now_str = now.strftime("%Y%m%d-%H%M%S")
file_dir = f'160906_pizza1-00001972-t5-p528/{now_str}'
os.makedirs(file_dir, exist_ok=True)

vis_color = {
    'gt':[255, 0, 0],
    'pd':[14, 83, 167],
    'link':[255, 201, 115],
    'gt-l':[255, 115, 115],
    'pd-l':[104, 153, 211],
}

def CFcv2(c):
    return vis_color[c][::-1]

def CFmpt(c):
    return [vis_color[c][0]/255,vis_color[c][1]/255,vis_color[c][2]/255]

def vis_2d_perpose_with_gt(image,joints,joints_vis,file_name):
    joint_num = joints_vis.shape[1]
    num_person= joints_vis.shape[0]
    
    alpha_line=0.7
    color_line=CFcv2('link')
    color_pt=[CFcv2('gt') , CFcv2('pd')]

    ndarr = image.permute(1, 2, 0).cpu().numpy()[...,::-1]
    ndarr = ndarr.copy()
    
    overlay = np.zeros(ndarr.shape, np.uint8)
    for i in range(joint_num):
        if joints_vis[0,i,0]:
            for p in range(1,num_person):
                cv2.line(overlay, (int(joints[0,i,0]), int(joints[0,i,1])),(int(joints[p,i,0]), int(joints[p,i,1])),color_line,2,cv2.LINE_AA)
    
    ndarr = cv2.addWeighted(ndarr, 1, overlay, alpha_line, 0 ,dtype=32)
    for n in range(num_person):
        overlay = np.zeros(ndarr.shape, np.uint8)
        for joint, joint_vis in zip(joints[n], joints_vis[n]):
            if joint_vis[0]:
                cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 3, color_pt[n%2], -1, cv2.LINE_AA)
    
    cv2.imwrite(file_name, ndarr)



LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
           [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]


def vis_3d_perpose_with_gt(gt_joints,gt_joints_vis,pred_joints,pred_vis,file_name,target_id=-1,ax=None):

    if ax == None:
        plt.axis('equal')
        plt.figure(0, figsize=(9, 9))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                            top=0.95, wspace=0.05, hspace=0.15)
        ax = plt.subplot(1, 1, 1, projection='3d')
    
    for n in range(gt_joints.shape[0]):
        joint = gt_joints[n]
        joint_vis = gt_joints_vis[n]
        for v in eval("LIMBS{}".format(len(joint))):
            x = [float(joint[v[0], 0]), float(joint[v[1], 0])]
            y = [float(joint[v[0], 1]), float(joint[v[1], 1])]
            z = [float(joint[v[0], 2]), float(joint[v[1], 2])]

            color = CFmpt('gt') if n==target_id else CFmpt('gt-l')
            if joint_vis[v[0]]:
                ax.plot(x, y, z, c=color, lw=2, marker='o',markerfacecolor='w', markersize=2,markeredgewidth=1)
            else:
                ax.plot(x, y, z, c=color, ls='--', lw=2, marker='o', markerfacecolor='w', markersize=2,markeredgewidth=1)
    
    ax.view_init(elev=75, azim=45)
    plt.title(file_name.split('/')[-1].split('.')[-2])
    plt.savefig(file_name,bbox_inches='tight',pad_inches = 0)
    plt.close(0)

CAM_LIST={
    'CMU0_ori': [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)],  # Origin order in MvP
    'CMU0' : [(0, 3), (0, 6),(0, 12),(0, 13), (0, 23)],
    'CMU0ex' : [(0, 3), (0, 6), (0, 12),(0, 13), (0, 23), (0, 10), (0, 16)],
    'CMU1' : [(0, 1),(0, 2),(0, 3),(0, 4),(0, 6),(0, 7),(0, 10)],  
    'CMU2' : [(0, 12), (0, 16), (0, 18), (0, 19), (0, 22), (0, 23), (0, 30)],
    'CMU3': [(0, 10), (0, 12), (0, 16), (0, 18)],
    'CMU4' : [(0, 6), (0, 7), (0, 10), (0, 12), (0, 16), (0, 18), (0, 19), (0, 22), (0, 23), (0, 30)],
}


def vis_cam(cam_seq):
    visualizer = CameraPoseVisualizer([-4000, 4000], [-4500, 3500], [0, 2800])
    cam_dict = np.load('cam_dict.npy',allow_pickle=True).item()['160422_ultimatum1']
    cam_list = [(cam_index[1],cam_dict[cam_index]) for cam_index in CAM_LIST[cam_seq]]
    for (cam_index,cam) in cam_list:
        our_cam = {}
        our_cam['R'] = cam['R']
        our_cam['T'] = -np.dot(cam['R'].T, cam['t']) * 10.0  # cm to mm
        our_cam['standard_T'] = cam['t'] * 10.0
        our_cam['fx'] = np.array(cam['K'][0, 0])
        our_cam['fy'] = np.array(cam['K'][1, 1])
        our_cam['cx'] = np.array(cam['K'][0, 2])
        our_cam['cy'] = np.array(cam['K'][1, 2])
        our_cam['k'] = cam['distCoef'][[0, 1, 4]].reshape(3, 1)
        our_cam['p'] = cam['distCoef'][[2, 3]].reshape(2, 1)
        our_cam['ex'] = np.concatenate((np.concatenate((our_cam['R'].T,our_cam['T']),1),np.array([0,0,0,1]).reshape(1,4)),0)

        visualizer.ax.scatter(our_cam['T'][0],our_cam['T'][1],our_cam['T'][2],c=matplotlib.cm.rainbow(cam_index / 30))

        visualizer.extrinsic2pyramid(our_cam['ex'], matplotlib.cm.rainbow(cam_index / 30), 1500)
        
    
    tmp_dict = torch.load('160906_pizza1-00001972-t5-p528.pth',map_location=torch.device('cpu'))
    

    for k,v in tmp_dict.items():
        if '3d-l3' in k:
            gt_joints = v['gt_joints']
            gt_vis = v['gt_vis']
            pred_joints = v['pred_joints']
            pred_vis = v['pred_vis']
            mpjpe = v['mpjpe']
            file_name = f'{file_dir}/{cam_seq}.jpg'
            vis_3d_perpose_with_gt(gt_joints,gt_vis,pred_joints,pred_vis,file_name,target_id=-1,ax=visualizer.ax)
    
    # visualizer.show(cam_seq)


if __name__ == '__main__':
    vis_cam('CMU0')
    vis_cam('CMU1')
    vis_cam('CMU2')
    vis_cam('CMU3')
    vis_cam('CMU4')
    vis_cam('CMU0ex')

