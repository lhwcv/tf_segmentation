## Here we implement FCN (Under going!)
   Please refer the paper [FCN](https://arxiv.org/abs/1411.4038)</br>
![](https://github.com/lhwcv/tf_segmentation/tree/master/FCN/tmp/net.png)
   

### 一些参考
   [知乎Deconvolutions 的讨论](https://www.zhihu.com/question/43609045?sort=created)</br>
   [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf)</br>
   [Is the deconvolution layer the same as a convolutional layer?](https://arxiv.org/ftp/arxiv/papers/1609/1609.07009.pdf)</br>
   [conv arithmetic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html)</br>
   [caffe 中对卷积的优化](http://lib.csdn.net/article/aiframework/62849)</br>
   [矩阵方式实现卷积](https://buptldy.github.io/2016/10/01/2016-10-01-im2col/)</br>
   
   上述参考主要是介绍了什么是transpose  convolution, 一方面从矩阵转置的角度阐述</br>
   另一方面从fractional strided 的角度： 填充些数值，然后转化为卷积操作。</br>
   第而二个方面可用于图形理解。 实际做的过程中是基于矩阵的，分两种： 一种是将kernel进行变换，</br>
   （采用Toeplitz matrix）变换为矩阵C，C进行转置, 而将图片进行简单reshape 得到X,  C 和 X 相乘</br>
   是卷积forward的过程, 而C的转置与X相乘是Deconvolution  的forward过程（也是卷积的backward过程）</br>
   另一种则是变换图像， 而kernel只做简单reshape, 图像的变换为 im2col  的过程， 需要一个col2im 的</br>
   反变换来完成transpose conv  或者 conv 的backward。</br>
   
   下面的一些参考是辅助：
   [Toeplitz matrix](http://blog.csdn.net/lanchunhui/article/details/72190213)</br>
   [卷积转为矩阵相乘，通过变换kernel](http://blog.csdn.net/lingerlanlan/article/details/23863347)</br>
   [kernel 核进行bilinear初始化](http://blog.csdn.net/jiongnima/article/details/78578876)</br>
   [bilinear 核初始化可用resize去代替](https://distill.pub/2016/deconv-checkerboard/)</br>
   [双线性插值](https://en.wikipedia.org/wiki/Bilinear_interpolation)</br>
   [用numpy实现caffe方式的卷积forward 和backward](https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/)</br>
   [Upsample with tensorflow](https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/)
### 我的感受及问题
   FCN 这篇算是比较基础的思路， 融合不同层级特征以获得更好的context信息，没有深度decoder</br>
   的过程， 直接上采样（通过 trans_conv）的到分割map, 核心大概是在这里的trans_conv 了， 而在</br>
   早期也已经有人提出来用其做upsample，  把它放在fcn中，我的理解就是一个resize， 事实上改成resize也完全</br>
   work 。  而这里bilinear  核的初始化仍然让我有点困惑， 不知道为什么这么算的，不过应该可以暂时忽略。
   

