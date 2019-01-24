# Implementing SIFT

This is an implementation of SIFT (Scale Invariant Feature Transform) from the paper *Distinctive Image Features from Scale-Invariant Keypoints (Lowe, 2004)*, using C++. OpenCV is used for its image container as well as some basic image processing subroutines.

### Usage
Currently supports matching between 2 images.

- To create Makefile, run
```bash=
cmake .
```

- To compile, run
```bash=
make
```

- To match, run
```bash=
./sift <image1> <image2>
```

### Sample Result
![](https://i.imgur.com/PatXdOi.png)

### Implementation Details
This implementation stays mostly true to that of Lowe's. The following is a summary of the procedure, for details please refer to the paper. (see references)

1. Scale space: 
Difference of Gaussian of input image under different $\sigma$, called "scale", is used to aproximate the scale-normalized Laplacian of Gaussian. $\sigma=1.6$ is set according to paper.
![](https://i.imgur.com/wFPIVIB.jpg)

3. Scale space extrema:
Local maxima and minima in the scale space are selected as candidates for features.
4. Low contrast rejection
$\hat x=-\frac{\partial^2D}{\partial x^2}^{-1}\frac{\partial D}{\partial x}$
$D(\hat x)=D+0.5\frac{\partial D}{\partial x}^T \hat x$
Accurate keypoint localization is calculated by fitting a 3D quadratic function to the local sample points. Then the DoG response at the new extrema is checked for value. Points with $|D(\hat x)|<0.03 \times 255$ are discarded. 
![](https://i.imgur.com/khO4XGk.jpg)


5. Edge response elimination
Peaks in difference of Gaussian will have large principle curvatures across the edge but a small one in the perpendicular direction. The ratio of principle curvatures $r$ can be calculated from 2^nd^ order derivatives, which is precomputed for the image.
$\frac{(D_{xx}+D_{yy})^2}{D_{xx}D_{yy}-(D_{xy})^2}=\frac{(r+1)^2}{r}<\frac{(r_0+1)^2}{r_0}$
According to the paper, $r_0=10$ is used.
![](https://i.imgur.com/OSBe58b.jpg)

6. Orientation assignment
Gradient orientation of Gaussian image around a keypoint is used to assign orientation to the keypoint. Gradient magnitude is used for weighting.
![](https://i.imgur.com/vAl8TY7.jpg)

7. Creating descriptors
- Gradient orientation is used again for creation of keypoint descriptor. 
- However, the orientations are shifted relative to the orientation of the keypoint. 
- The orientations are divided into 8 bins and added up across a 4x4 sub-region. 
- According to the paper, 16 of such descriptor are calculated, ceating a 8x16=128 dimension vector for each of the keypoints found. 
- The vectors are normalized to unit length.
![](https://i.imgur.com/2APmqU0.png)
(image taken from the paper)

8. Descriptor matching
- For now, an exhaustive is used to match the features. 
- The paper uses euclidean distance between vectors to determine a match. Only the matches in which the ratio of closest distance to second closest one is lesser than 0.8 are viewed as correct match.
In other words, $\frac{v - v_1}{v - v_2} < 0.8, \forall v_2$
- To simplify the calculation, we use inner product instead. More specifically, $\frac{1-v^Tv_1}{1-v^Tv_2}< 0.64$
- To derive the above result, we simply square the original equation, coupled with the fact that all vectors are unit-lengthed:
$(\frac{v - v_1}{v - v_2})^2 < 0.8^2\to \frac{1-v^Tv_1}{1-v^Tv_2}< 0.64$

### References
- Distinctive Image Features from Scale-Invariant Keypoints. Lowe, 2004.
- OpenCV 4
