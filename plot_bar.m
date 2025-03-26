clc
clear
figure()
tps = [645.8 869.5 859.4; 1252.3 1510.0 1245.9; 1651.5 2820.4 2740.4];
qps = [12923.0 17395.8 17198.9; 20036 24158.4 19935.3; 9909.2 16922.3 16441.2];
lat = [651.9 541.7 586.5; 268.8 192.0 278.4; 262.6 175.9 201.4];
tpmC = [16632.1 19021.3 19180.3];

Method{1} = 'oltp\_read\_write';
Method{2} = 'oltp\_read\_only';
Method{3} = 'oltp\_write\_only';
Method{4} = 'TPC-C';

b = bar(tpmC, 0.6); % 0.6表示宽度占比
set(gca, 'FontSize', 13)
set(gcf, 'Position', [100 100 850 600]); % 增加图形的高度，这里将高度从600改为800
ylabel('95% latency(ms)', 'FontSize', 14) % y轴坐标描述
set(gca, 'xticklabel', {Method{1}, Method{2}, Method{3}, Method{4}})
% ylim([0,10]);
set(gca, 'XTick', [1:1:4])

% 在蓝色的柱状图的上，显示数值
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints; % 获取 Bar 对象的 XEndPoints 和 YEndPoints 属性
labels1 = string(b(1).YData); % 获取条形末端的坐标
text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'bottom')

% 在黄色的柱状图的上，显示数值
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints; % 获取 Bar 对象的 XEndPoints 和 YEndPoints 属性
labels2 = string(b(2).YData); % 获取条形末端的坐标
text(xtips2, ytips2, labels2, 'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'bottom')

% 在红色的柱状图的上，显示数值
xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints; % 获取 Bar 对象的 XEndPoints 和 YEndPoints 属性
labels3 = string(b(3).YData); % 获取条形末端的坐标
text(xtips3, ytips3, labels3, 'HorizontalAlignment', 'center',...
    'VerticalAlignment', 'bottom')

grid on % 打开网格
set(gca, 'XTickLabelRotation', 0); % 设置横坐标不倾斜
set(gca, 'YGrid', 'on'); % 设置y轴刻度有虚线

% 添加图例
legend('默认参数', 'DDPG', 'MADDPG', 'Location', 'northeast');

% 保存图形到本地并指定分辨率
print('./img/bar_plot', '-dpng', '-r500'); % 500dpi的分辨率

%% TPC-C
clc
clear
figure()

tpmC = [16632.1 19021.3 19180.3];

Method{1} = '默认参数';
Method{2} = 'DDPG';
Method{3} = 'MADDPG';

% 创建与 tpmC 大小相同的位置数组
x = 1:numel(tpmC);

% 定义颜色
colors = {'b', 'r', 'y'};

% 绘制柱状图
figure;
hold on;
for i = 1:numel(tpmC)
    b(i) = bar(x(i), tpmC(i), 'FaceColor', colors{i});
end
hold off;

% 设置柱子的宽度
for i = 1:numel(b)
    b(i).BarWidth = 0.35; % 设置柱子宽度为默认的一半
end

% 设置柱子之间的空隙
barWidth = 0.5;
barSpacing = 0.2; % 柱子之间的空隙
totalWidth = (numel(tpmC) - 1) * barSpacing + numel(tpmC) * barWidth;
firstX = 0; % 将柱子整体向左移动一些距离
for i = 1:numel(tpmC)
    b(i).XData = firstX + (i - 1) * (barWidth + barSpacing);
end

% 设置字体大小
set(gca, 'FontSize', 13)

% 设置图形大小
set(gcf, 'Position', [100 100 700 500]); % 将宽度从850改为700

% 设置 y 轴标签
ylabel('每秒交易数（tpmC）', 'FontSize', 14)

% 设置 x 轴刻度标签
set(gca, 'xticklabel', Method)

% 设置 x 轴刻度
set(gca, 'XTick', firstX + (0:numel(tpmC)-1) * (barWidth + barSpacing))

% 在柱子顶部显示数值，并使其与柱子中心对齐
for i = 1:numel(b)
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    labels = string(b(i).YData);
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% 打开网格
grid on

% 设置 x 轴刻度标签不倾斜
set(gca, 'XTickLabelRotation', 0);

% 设置 y 轴刻度有虚线
set(gca, 'YGrid', 'on');

% 添加图例
legend(Method, 'Location', 'northeast');

% 保存图形到本地并指定分辨率
print('./img/bar_plot.png', '-dpng', '-r500'); % 500dpi的分辨率

%% 饼图


