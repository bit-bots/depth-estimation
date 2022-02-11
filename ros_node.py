import rospy
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

MODEL_PATH="checkpoints/checkpoint_epoch5.pth"

class DepthEstimator:
    def __init__(self):
        rospy.init_node('depth_estimator')
        self.depth_pub = rospy.Publisher("/image_depth_estimation", Image, queue_size = 1)
        self.subscriber = rospy.Subscriber("/camera/image_proc", Image, self.callback, queue_size = 1)
        self.cv_bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = UNet(n_channels=3)
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.net.eval()
        rospy.spin()

    def callback(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        shape = img.shape
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = torch.from_numpy(BasicDataset.preprocess(pil_img, 0.2, is_mask=False))
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.net(img)
            output = -1/(1-(1/output))

            tf = transforms.Compose([
                transforms.Resize((shape[0], shape[1])),
            ])

            output = tf(output).squeeze().cpu().numpy()[..., np.newaxis]

        print(output.shape, output.dtype)
        depth_msg = self.cv_bridge.cv2_to_imgmsg(output, "32FC1")
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)

if __name__ == "__main__":
    DepthEstimator()
