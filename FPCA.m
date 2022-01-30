% Jaime Zabalza, Jinchang Ren, et al., Novel folded-PCA for improved feature extraction 
% and data reduction with hyperspectral imaging and SAR in remote sensing, 
% ISPRS Journal of Photogrammetry and Remote Sensing, 2014

function [M]=FPCA(comp,H,D)

[len,pixels]=size(D);
compH=comp/H;   % number of components per row
ss=len/H; % row size (number of columns)
        
[len,N]=size(D);
u=mean(D.').';
D=D-repmat(u,1,N);
   
% covariance matrix
C=zeros(ss,ss);     
for i=1:pixels   
    pix=D(:,i);
    p(:,:,i)=transform2matrix(pix,len,H,ss);    % from vector to matrix
    C=C+p(:,:,i).'*p(:,:,i);
end
C=C/pixels;

% EVD
[V,E]=eig(C);
%%%%%%%%%%%%%%%%%%%%%%%%%
E=diag(E);
[~,ind]=sort(E,'descend');
E=E(ind(1:compH));   % sorted Eigenvalues
V=V(:,ind(1:compH)); % sorted Eigenvectors
%%%%%%%%%%%%%%%%%%%%%%%%%    

% transformation
for i=1:pixels 
    pix_red(:,:,i)=p(:,:,i)*V;  % reduced matrix
end 
M=zeros(H*compH,pixels);
for k=1:pixels
    pix_red_total=[];   % reduced pixel in vector array   
    for i=1:H
        for j=1:compH
            pix_red_total=[pix_red_total pix_red(i,j,k)];
        end
    end
M(:,k)=pix_red_total;   % locating each pixel

end

end

function [p]=transform2matrix(pix,band,H,W)
l=length(pix);
p=zeros(H,W);   % matrix
n=1;
for i=1:H
    for j=1:W
        p(i,j)=pix(n);
        n=n+1;
    end
end
end
