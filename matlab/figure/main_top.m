%���ƣ�main_top
%���ܣ�������Ӱ��״ͼ������ͼ
%ע�ͣ�������ˮ
%��ע��������Ӱ��ʽ'/', '\', '|', '-', '+', 'x', '.'
%data��2018-0421

clear;
close all;
data=[40 9;24 10; 12 3];
bar(data,1);
title('Hello world','FontName','Times New Roman','FontSize',15);   %ͼ��
xlabel('Hey girl','fontsize',15,'FontName','Times New Roman');     %���������ݼ�����
ylabel('Hey man','fontsize',15,'FontName','Times New Roman');      %���������ݼ�����
axis([0 4 0 50]);                                		   %�޸ĺ����귶Χ
legend('AA','BB','Square');                      		   %�޸�ͼ��
set(gca,'XTickLabel',{'Img1','Img2','Img3'},'FontSize',15,'FontName','Times New Roman');    %�޸ĺ��������ơ�����
applyhatch(gcf,'\.x.');                          		   %������Ӱ��ʽ'/', '\', '|', '-', '+', 'x', '.'