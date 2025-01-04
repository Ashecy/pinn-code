% 定义 x 和 t 的取值范围
x0 = -5; x1 = 5;  % x 的范围
t0 = -0.5; t1 = 0.5;  % t 的范围

% 定义分辨率
x_points = 513;  % x 方向的点数
t_points = 401;  % t 方向的点数

% 生成网格
x = linspace(x0, x1, x_points);
t = linspace(t0, t1, t_points);
[X, T] = meshgrid(x, t);

% 计算 q(x,t)
q = 2 * sech(8 * T + 2 * X) .* exp(-2i * X + 1i);
% 分离实部虚部
u = real(q);
v = imag(q);

% 可视化结果（例如模值的二维图像）
figure;
surf(X, T, abs(q));  % 绘制模值的曲面图
xlabel('x'); ylabel('t'); zlabel('|q(x,t)|');
title('q(x,t) = 2 sech(8t+2x) e^{(-2ix+1)}');
shading interp;  % 平滑显示
colorbar;  % 添加颜色条
save('data/train-data.mat');