%名称：main_top
%功能：绘制阴影柱状图、条形图
%注释：冰三点水
%备注：设置阴影格式'/', '\', '|', '-', '+', 'x', '.'
%data：2018-0421

clear;
close all;
data=[40 9;24 10; 12 3];
bar(data,1);
title('Hello world','FontName','Times New Roman','FontSize',15);   %图名
xlabel('Hey girl','fontsize',15,'FontName','Times New Roman');     %横坐标内容及字体
ylabel('Hey man','fontsize',15,'FontName','Times New Roman');      %纵坐标内容及字体
axis([0 4 0 50]);                                		   %修改横坐标范围
legend('AA','BB','Square');                      		   %修改图例
set(gca,'XTickLabel',{'Img1','Img2','Img3'},'FontSize',15,'FontName','Times New Roman');    %修改横坐标名称、字体
applyhatch(gcf,'\.x.');                          		   %设置阴影格式'/', '\', '|', '-', '+', 'x', '.'