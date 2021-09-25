## 骨架

前一段时间在看折痕消除的一篇论文的时候提到了提取文字的骨架，就重新复习一下《数字图像处理》这本书中关于骨架的知识。

集合 $A$ 的骨架的 $S(A)$ 概念上很简单：

**若 $z$ 是 $S(A)$ 的一点，$D_z$ 是 $A$ 内以 $z$ 为圆心的最大圆盘，则不存在包含 $D_z$ 且位于 $A$ 内的更大圆盘，满足这些条件的圆盘 $D_z$ 称为最大圆盘**

若 $D_z$ 是一个最大圆盘，则它在两个或多个不同的位置与 $A$ 的边界接触。

<img src="./skeleton.png" style="zoom: 25%;" />

$A$ 的骨架可以用腐蚀和开运算来表示：

$$
S(A) = \bigcup_{k=0}^{K}S_k(A)= \bigcup_{k=0}^{K}(A\ominus kB)-(A\ominus kB)\circ B
$$

式中，$B$ 是一个结构元，$A\ominus kB$ 表示 $A$ 的连续 $k$ 次腐蚀，同时，$K$ 是 $A$ 被腐蚀为一个空集前的最后一个迭代步骤：

$$
K = max\{ k| A\ominus kB \ne \oslash \}
$$

`Python` 的 `skimage` 库中提供了可以直接调用的方法：

`skimage.morphology.skeletonize(image)`