为了运行这里的代码，你需要构建一个 Python 环境. 并且安装所需的包：numpy, scipy, matplotlib.

这里推荐使用 VScode + Python. 环境配置方法可以参考：https://zhuanlan.zhihu.com/p/584126712

在上述文章中，执行 `pip install numpy` 时，顺便执行 `pip install scipy` 和 `pip install matplotlib`。

环境配置完成后，在 vscode 里打开 `scipy_demo.ipynb` 并运行就行。


1.阅读：
- ad_main.pdf
- scipy_demon.pdf
- 自动微分.pdf

2. 运行
- scipy_demon.ipynb

3.作业：
- 【必须】求解 ad_main.pdf 中的 `最后的优化问题`，得到一条轨迹 [x0,x1,...,xN]，xk 维度为 5。
- 【加分】求解的时候用自己写的求解器，而不是掉包。
- 【加分】对状态方程加噪声，采用滚动优化的方式求解泊车问题，得到一条轨迹 [x0,x1,...,xN]，xk 维度为 5。
- 【加分】利用带噪声的状态方程，采用 PID 跟踪这一条轨迹。

