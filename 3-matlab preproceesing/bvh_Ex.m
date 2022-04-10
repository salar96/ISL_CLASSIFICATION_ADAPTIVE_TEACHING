clear
close all
clc
%%
path='C:\Users\Salar Basiri\Desktop\matsddddata new\';
listing=dir(path);
ls=size(listing,1)
%%
counter=1;
for i=3:ls 
    (i/ls)*100
    address=strcat(path,listing(i).name);
    %[skel,time]=loadbvh(strcat(path,listing(i).name));
    v=catch_and_aug(address);
    %w=add_noise(v);
    w=v;
    s=size(w,2);
    for n=1:s      
    out(counter,:,:)=norm_and_id(w{n},give_id(strcat(path,listing(i).name)));
    counter=counter+1;
    end
end

