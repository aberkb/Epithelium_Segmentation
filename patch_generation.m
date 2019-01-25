outdir='./patches/';
mkdir(strrep(outdir,'..','.'));
patchSz = [200 200];

imSz = [1000 1000];
xIdxs = [1:patchSz(2):imSz(2) imSz(2)+1];
yIdxs = [1:patchSz(1):imSz(1) imSz(1)+1];
patches = cell(length(yIdxs)-1,length(xIdxs)-1);
imagefiles = dir('*.tif');      
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   I = imread(currentfilename);
   imSz = size(I);
    for i = 1:length(yIdxs)-1
        Isub = I(yIdxs(i):yIdxs(i+1)-1,:);
        for j = 1:length(xIdxs)-1
            pat = Isub(:,xIdxs(j):xIdxs(j+1)-1);
            pname=sprintf('%d_e_%d___%s.png',i,j,currentfilename); %generate a filename for it
            imwrite(pat,sprintf('%s/%s',outdir,pname)); %save the image patch to the output directoy
        end

    end
end



