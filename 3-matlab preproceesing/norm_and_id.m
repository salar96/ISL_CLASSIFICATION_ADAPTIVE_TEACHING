function y= norm_and_id(x,id )
my_size=60;
ss=size(x,2);
t=round(linspace(1,ss,my_size));
xx=zeros(size(x,1),size(t,2));
for j=1:size(xx,1)
for p=1:size(t,2)
    if j==size(xx,1)-2 || j==size(xx,1)-1 || j==size(xx,1)
        xx(j,p)=round(x(j,t(p)),1);
    else
    xx(j,p)=round(x(j,t(p)));
    end
end
end
s=size(xx,1);
y=zeros(s,my_size);
for i=1:s
    M=max(xx(i,:));
    m=min(xx(i,:));
    y(i,:)=xx(i,:)*255/(M-m)-255*m/(M-m);
   %y(i,:)=xx(i,:);
end
y(1,end+1)=id;
end