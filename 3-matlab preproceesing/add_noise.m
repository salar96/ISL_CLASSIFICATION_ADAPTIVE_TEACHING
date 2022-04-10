function y = add_noise( x )
% v is a cell
s=size(x,2);
noise_num=1;
ss=s*noise_num;
y=cell(1,s);

counter=1;
for i=1:s
    
    w=x{i};
    nsatr=size(w,1);
    y{counter}=w;
    counter=counter+1;
    for n=1:noise_num
        w=x{i};
        if rand()<0.5
            for j=1:nsatr
             w(j,:)=awgn(w(j,:),rand()*5+35,'measured');
             end
          y{counter}=w;
          counter=counter+1;
        end
    end
end
end
