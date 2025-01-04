% Load the predicate data
data = load('output/predicate_data.mat');

% Extract data
X = data.X;
T = data.T;
real_q = data.real_q;
pred_q = data.pred_q;
error_q = data.error_q;

% Plot real data
figure;
contourf(X, T, real_q, 50, 'LineColor', 'none');
colorbar;
title('Real Data');
xlabel('x');
ylabel('t');
grid on; % or off

% Plot predicted data
figure;
contourf(X, T, pred_q, 50, 'LineColor', 'none');
colorbar;
title('Predicted Data');
xlabel('x');
ylabel('t');
shading interp;

% Plot error data
figure;
contourf(X, T, error_q, 50, 'LineColor', 'none');
colorbar;
title('Error Data');
xlabel('x');
ylabel('t');
shading interp;

% Plot 3d real data
figure;
surf(X, T, real_q, 'EdgeColor', 'none');
colorbar;
title('Real Data 3d');
xlabel('x');
ylabel('t');
zlabel('|q|');
shading interp;

% Plot 3d predicted data
figure;
surf(X, T, pred_q, 'EdgeColor', 'none');
colorbar;
title('Predicted Data 3d');
xlabel('x');
ylabel('t');
zlabel('|q|');
shading interp;