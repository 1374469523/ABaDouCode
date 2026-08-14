# ============================================================
# 演示主题：图像的加载与显示（用 matplotlib 读取并展示一张图片）
#
# 核心结论：
#   1. plt.imread() 可以把图片文件读取成一个 numpy 数组，方便后续交给
#      PyTorch 做卷积等运算；
#   2. 图片数组的形状是 (高H, 宽W, 通道C)：彩色图 C=3（RGB 三通道），
#      灰度图 C=1。这就是后面 img.shape 输出形如 (H, W, 3) 的原因；
#   3. plt.imshow() 负责把像素数组渲染成图像，plt.show() 负责弹出窗口显示。
# ============================================================
import matplotlib.pyplot as plt  # 导入 matplotlib 绘图库，用于显示图片
import numpy as np               # 导入 numpy 数组库，用于表示和处理图片数据

# 下面是被注释掉的示例：演示如何手动生成纯色图片。
# np.zeros([200,200,3])：生成 200x200、3 通道、所有像素值都为 0 的图片 → 纯黑
# img1 =np.zeros([200,200,3])
# plt.imshow(img1)
# plt.show()
#
# np.full([200,200,3],128)：生成所有像素值都为 128 的图片 → 灰色
# img2 =np.full([200,200,3],128)
# plt.imshow(img2)
# plt.show()

# plt.imread(路径)：读取图片文件，返回 numpy 数组，每个元素是一个像素的 RGB 值
img = plt.imread('/Users/mac/Desktop/AI20深度学习/02-code/03-CNN/data/img.jpg')
print(img.shape)        # 打印图片数组形状：(高H, 宽W, 3)，3 代表 RGB 三个颜色通道
plt.imshow(img)         # 把像素数组绘制到画布上（渲染成图片）
plt.show()              # 弹出窗口显示图片（不加这行，图像不会显示出来）
