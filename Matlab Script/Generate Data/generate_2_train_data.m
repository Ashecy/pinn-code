 % 定义 x 和 t 的取值范围
x0 = -5; x1 = 5;  % x 的范围
t0 = -3; t1 = 3;  % t 的范围

% 定义分辨率
x_points = 513;  % x 方向的点数
t_points = 513;  % t 方向的点数

% 生成网格
x = linspace(x0, x1, x_points);
t = linspace(t0, t1, t_points);
[X, T] = meshgrid(x, t);

% 定义 A 和 B

A = (0.224 + 0.158i)*exp(1i*(1.92*T + 0.32i*T - 0.02*X - 0.8i*X)) - (0.024 + 0.232i)*exp(-1i*(-0.6*T + 0.56i*T - 0.2*X + 1.4i*X))...
    + (0.024 - 0.232i)*exp(1i*(0.6*T + 0.56i*T + 0.2*X + 1.4i*X)) + (0.518i - 0.224)*exp(-1i*(-1.92*T + 0.32i*T + 0.2*X - 0.8i*X));

B = 1.25*exp(0.88*T + 0.6*X) + 0.13*exp(-0.24*T - 2.2*X) + 1.25*exp(-0.88*T - 0.6*X)...
    - 1.22*exp(1i*(1.32*T - 0.4*X)) - 1.12*exp(-1i*(1.32*T - 0.4*X)) + 0.13*exp(0.24*T + 2.2*X);



% 定义 q(x, t)
q = -2i * A ./ B;
% 分离实部虚部
u = real(q);
v = imag(q);

% 可视化结果（例如模值的二维图像）
figure;
surf(X, T, abs(q));  % 绘制模值的曲面图
xlabel('x'); ylabel('t'); zlabel('|q(x,t)|');
title('two soliton');
shading interp;  % 平滑显示
colorbar;  % 添加颜色条
%save('data/2-train-data.mat');

