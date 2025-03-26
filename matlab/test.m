clc
close all
clear
%% 基本设置
pathFigure= '.\' ;

%% Example 1：柱状图填充

figure(1);
h = bar(rand(3,4));
xlabel('Xlabel','fontsize',14,'FontName','Times New Roman','FontWeight','Bold')
ylabel('Ylabel','fontsize',14,'FontName','Times New Roman','FontWeight','Bold')
set(gca,'Layer','top','FontSize',14,'Fontname', 'Times New Roman');

str= strcat(pathFigure, "Figure1", '.tiff');
print(gcf, '-dtiff', '-r600', str);

figure(2);
hp = bar(rand(3,4));
xlabel('Xlabel','fontsize',14,'FontName','Times New Roman','FontWeight','Bold')
ylabel('Ylabel','fontsize',14,'FontName','Times New Roman','FontWeight','Bold')
set(gca,'Layer','top','FontSize',14,'Fontname', 'Times New Roman');
hatchfill2(hp(1),'single','HatchAngle',0);
hatchfill2(hp(2),'cross','HatchAngle',45);
hatchfill2(hp(3),'single','HatchAngle',90);

str= strcat(pathFigure, "Figure2", '.tiff');
print(gcf, '-dtiff', '-r600', str);

%%
% 清理环境
clc;
clear;
figure();

% 示例数据
lat = [651.9 541.7 586.5; 268.8 192.0 278.4; 262.6 175.9 201.4];

Method{1} = 'oltp_read_write';
Method{2} = 'oltp_read_only';
Method{3} = 'oltp_write_only';

% 创建白色柱状图
b = bar(lat, 0.6, 'w'); % 使用 'w' 参数来设置为白色
set(gca, 'FontSize', 13)
set(gcf, 'Position', [100 100 850 600]); % 增加图形的高度，这里将高度从600改为800
ylabel('95% latency(ms)', 'FontSize', 14) % y轴坐标描述
set(gca, 'xticklabel', {Method{1}, Method{2}, Method{3}})
set(gca, 'XTick', [1:1:3])

% 添加花纹
hatchStyles = {'single', 'cross', 'single'};
hatchAngles = [0, 45, 90];
hatchDensities = [20, 10, 15];

for k = 1:size(lat, 2)
    hatchfill2(b(k), 'HatchStyle', hatchStyles{k}, 'HatchAngle', hatchAngles(k), 'HatchDensity', hatchDensities(k), 'HatchColor', 'k');
end

% 在柱状图的上方显示数值
for i = 1:length(b)
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints; % 获取 Bar 对象的 XEndPoints 和 YEndPoints 属性
    labels = string(b(i).YData); % 获取条形末端的坐标
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
end

% 打开网格
grid on;
set(gca, 'XTickLabelRotation', 0); % 设置横坐标不倾斜
set(gca, 'YGrid', 'on'); % 设置y轴刻度有虚线

% 添加图例
legend({'默认参数', 'DDPG', 'MADDPG'}, 'Location', 'northeast');

% 保存图形到本地并指定分辨率
pathFigure = './';
str = strcat(pathFigure, "Figure2", '.tiff');
print(gcf, '-dtiff', '-r600', str);




