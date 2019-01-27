outdir='./patches/';
mkdir(strrep(outdir,'..','.'));
patchSize = [200 200];

imgSize = [1000 1000];
xIndex = [1:patchSize(2):imgSize(2) imgSize(2)+1];
yIndex = [1:patchSize(1):imgSize(1) imgSize(1)+1];
patches = cell(length(yIndex)-1,length(xIndex)-1);
imagefiles = dir('*.tif');
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
   currentFileName = imagefiles(ii).name;
   I = imread(currentFileName);
   imgSize = size(I);
    for i = 1:length(yIndex)-1
        Isub = I(yIndex(i):yIndex(i+1)-1,:);
        for j = 1:length(xIndex)-1
            pat = Isub(:,xIndex(j):xIndex(j+1)-1);
            pname=sprintf('%d_e_%d___%s.png',i,j,currentfilename); %generate a filename for each img
            imwrite(pat,sprintf('%s/%s',outdir,pname)); %save the image patch to the output directoy
        end
    end
end
