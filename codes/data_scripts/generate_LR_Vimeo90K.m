function generate_LR_Vimeo90K()
%% matlab code to genetate bicubic-downsampled for Vimeo90K dataset

up_scale = 4;
mod_scale = 4;
idx = 0;
filepaths = dir('/media/data3/wh/data/dataset/SR/vimeo_septuplet/sequences/train/*/*/*.png');
for i = 1 : length(filepaths)
    [~,imname,ext] = fileparts(filepaths(i).name);
    folder_path = filepaths(i).folder;
    save_LR_folder = strrep(folder_path,'vimeo_septuplet','vimeo_septuplet_matlabLRx4_BD_trainm');
    if ~exist(save_LR_folder, 'dir')
        mkdir(save_LR_folder);
    end
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_rlt = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_rlt);
        % read image
        img = imread(fullfile(folder_path, [imname, ext]));
        img = im2double(img);
        % modcrop
        img = modcrop(img, mod_scale);
        % LR
        %im_LR = imresize(img, 1/up_scale, 'bicubic');
        im_LR = imresize_BD(img, up_scale, 7, 1.6);
        if exist('save_LR_folder', 'var')
            imwrite(im_LR, fullfile(save_LR_folder, [imname, '.png']));
        end
    end
end
end

%% modcrop
function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2),:);
end
end


%% BD downsampling
function ImLR = imresize_BD(ImHR, scale, kernelsize, sigma)
% ImLR and ImLR are
% downsample by Bicubic
kernel = fspecial('gaussian', kernelsize, sigma);
blur_HR = imfilter(ImHR, kernel, 'replicate');
ImLR = imresize(blur_HR, 1/scale, 'nearest');

end
