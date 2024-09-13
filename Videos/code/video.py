import os
import cv2

# 设定起始和结束的图片名
start_index = '2942.jpg'
end_index = '3692.jpg'

# 图片文件夹路径
image_folder = '/home2/wzc/UniAD/Videos/images/'

# 获取图片名列表
image_names = sorted(os.listdir(image_folder))
start_index_found = False
images = []

# 遍历图片名列表
for image_name in image_names:
    if image_name == start_index:
        start_index_found = True

    if start_index_found:
        images.append(image_name)

    if image_name == end_index:
        break

# 获取第一张图片的分辨率
first_image_path = os.path.join(image_folder, images[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape

# 创建视频文件
output_video = f"{start_index.split('.')[0]}_{end_index.split('.')[0]}.mp4"
output_path = os.path.join('/home2/wzc/UniAD/Videos/mp4_version/', output_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_path, fourcc, 10, (width, height))

# 逐个读取图片并写入视频
for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    video.write(image)

# 释放资源
video.release()
cv2.destroyAllWindows()