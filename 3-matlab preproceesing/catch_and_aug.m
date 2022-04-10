function v = catch_and_aug(filename)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % forearm
% x_forearm=skel(19).Dxyz(1,:);y_forearm=skel(19).Dxyz(2,:);z_forearm=skel(19).Dxyz(3,:);
% % hand
% x_hand=skel(20).Dxyz(1,:);y_hand=skel(20).Dxyz(2,:);z_hand=skel(20).Dxyz(3,:);
% % thumb
% x_thumb=skel(23).Dxyz(1,:);y_thumb=skel(23).Dxyz(2,:);z_thumb=skel(23).Dxyz(3,:);
% % index
% x_index=skel(28).Dxyz(1,:);y_index=skel(28).Dxyz(2,:);z_index=skel(28).Dxyz(3,:);
% % middle
% x_middle=skel(33).Dxyz(1,:);y_middle=skel(33).Dxyz(2,:);z_middle=skel(33).Dxyz(3,:);
% % ring
% x_ring=skel(38).Dxyz(1,:);y_ring=skel(38).Dxyz(2,:);z_ring=skel(38).Dxyz(3,:);
% % pinky
% x_pinky=skel(43).Dxyz(1,:);y_pinky=skel(43).Dxyz(2,:);z_pinky=skel(43).Dxyz(3,:);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% w=[x_forearm;y_forearm;z_forearm;x_hand;y_hand;z_hand;x_thumb;y_thumb;z_thumb;x_index;y_index;z_index;x_middle;y_middle;z_middle;x_ring;y_ring;z_ring;x_pinky; y_pinky;z_pinky];
w=readNPY(filename);
s1=size(w,1);
s2=size(w,2);
counter=1;
 %% adding delay and truncation
%  v=cell(1,100);
i_min=-10;
i_max=20;
j_min=-10;
j_max=20;
step_i=5;
step_j=5;
vtx=((i_max-i_min)/step_i+1)*((j_max-j_min)/step_j+1);
v=cell(1,vtx);
 for i=i_min:step_i:i_max % i represents the begining
     for j=j_min:step_j:j_max % j represents the end
         
         if i>0
            k=round(abs(i)/100*s2);
            u=zeros(s1,k);
            for p=1:k
                u(:,p)=w(:,1);
            end
            v{counter}=[u,w];
           
         end
         
         if i==0
             u=[];
            v{counter}=[u,w];
            
         end
         
         if i<0
            k=round(abs(i)/100*s2);
            O=w;
            O(:,1:k)=[];
            v{counter}=O;
         end
         
         if j>0
            k=round(abs(j)/100*s2);
            u=zeros(s1,k);
            for p=1:k
                u(:,p)=w(:,end);
            end
            v{counter}=[v{counter},u];
         end
         
         if j==0
            u=[];
            v{counter}=[v{counter},u];
         end
         
         if j<0
            k=round(abs(j)/100*s2);
            m=v{counter};
            m(:,end-k:end)=[];
            v{counter}=m;  
         end
         counter=counter+1;
     end
%      counter=counter+1;
 end

 %%%%
 k=round(abs(17)/100*s2);
 u=zeros(s1,k);
            for p=1:k
                u(:,p)=w(:,1);
            end
            v{vtx+1}=[u,w];
%%%%
end

