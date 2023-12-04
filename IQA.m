% Add paths of all IQA algorithms
addpath('E:\PyCharmProjects\dippykit\CSV-master\Code');
addpath('E:\PyCharmProjects\dippykit\CSV-master\Code\FastEMD');
addpath('E:\PyCharmProjects\dippykit\MS-UNIQUE-master');
addpath('E:\PyCharmProjects\dippykit\MS-UNIQUE-master\InputWeights');
addpath('E:\PyCharmProjects\dippykit\UNIQUE-Unsupervised-Image-Quality-Estimation-master');
addpath('E:\PyCharmProjects\dippykit\UNIQUE-Unsupervised-Image-Quality-Estimation-master\InputWeights');
addpath('E:\PyCharmProjects\dippykit\SUMMER-master\Code');

% To change the category, change the directory before 'DSLR_JPG'
% Add path of original images
source_path = 'E:\PyCharmProjects\dippykit\10_grayscale_no_challenge\white\DSLR_JPG';

% Add path of bilinear, CABI KMeans, and CABI GMM upsampled images
bilinear_path = 'E:\PyCharmProjects\dippykit\upsample\white\bilinear';
kmeans_path = 'E:\PyCharmProjects\dippykit\upsample\white\kmeans';
gmm_path = 'E:\PyCharmProjects\dippykit\upsample\white\gmm';
source_files = dir(source_path);
bilinear_files = dir(bilinear_path);
kmeans_files = dir(kmeans_path);
gmm_files = dir(gmm_path);

% Create empty vectors to store scores of each IQA algorithm of each
% interpolation algorithm
PSNR_bilinear = [];
SSIM_bilinear = [];
CSV_bilinear = [];
MSUnique_bilinear = [];
CWSSIM_bilinear = [];
Unique_bilinear = [];
Summer_bilinear = [];

PSNR_kmeans = [];
SSIM_kmeans = [];
CSV_kmeans = [];
MSUnique_kmeans = [];
CWSSIM_kmeans = [];
Unique_kmeans = [];
Summer_kmeans = [];

SSIM_gmm = [];
PSNR_gmm = [];
CSV_gmm = [];
MSUnique_gmm = [];
CWSSIM_gmm = [];
Unique_gmm = [];
Summer_gmm = [];

% Load one original file and the upsampled image produced by the three
% algorithms
for i=3:length(source_files)
    % Load original grayscale image
    % Create three channels by copying the image into each channel
    im_source = imread(fullfile(source_path, source_files(i).name)); 
    im_source3 = cat(3, im_source, im_source, im_source);

    % Load bilinear grayscale image
    % Create three channels by copying the image into each channel
    im_bilinear = imread(fullfile(bilinear_path, bilinear_files(i).name));
    im_bilinear3 = cat(3, im_bilinear, im_bilinear, im_bilinear);

    % Load CABI Kmeans grayscale image
    % Create three channels by copying the image into each channel
    im_kmeans = imread(fullfile(kmeans_path, kmeans_files(i).name));
    im_kmeans3 = cat(3, im_kmeans, im_kmeans, im_kmeans);

    % Load CABI GMM grayscale image
    % Create three channels by copying the image into each channel
    im_gmm = imread(fullfile(gmm_path, gmm_files(i).name));
    im_gmm3 = cat(3, im_gmm, im_gmm, im_gmm);
    
    % Calculate CSV score of the three algorithms
    CSV_bilinear = [CSV_bilinear, csv(im_source3, im_bilinear3)];
    CSV_kmeans = [CSV_kmeans, csv(im_source3, im_kmeans3)];
    CSV_gmm = [CSV_gmm, csv(im_source3, im_gmm3)];

    % Calculate PSNR score of the three algorithms
    PSNR_bilinear = [PSNR_bilinear, psnr(im_source3, im_bilinear3)];
    PSNR_kmeans = [PSNR_kmeans, psnr(im_source3, im_kmeans3)];
    PSNR_gmm = [PSNR_gmm, psnr(im_source3, im_gmm3)];

    % Calculate SSIM score of the three algorithms
    SSIM_bilinear = [SSIM_bilinear, ssim(im_source3, im_bilinear3)];
    SSIM_kmeans = [SSIM_kmeans, ssim(im_source3, im_kmeans3)];
    SSIM_gmm = [SSIM_gmm, ssim(im_source3, im_gmm3)];

    % Calculate MS-UNIQUE score of the three algorithms
    MSUnique_bilinear = [MSUnique_bilinear, mslMSUNIQUE(im_source3, im_bilinear3)];
    MSUnique_kmeans = [MSUnique_kmeans, mslMSUNIQUE(im_source3, im_kmeans3)];
    MSUnique_gmm = [MSUnique_gmm, mslMSUNIQUE(im_source3, im_gmm3)];

    % Calculate UNIQUE score of the three algorithms
    Unique_bilinear = [Unique_bilinear, mslUNIQUE(im_source3, im_bilinear3)];
    Unique_kmeans = [Unique_kmeans, mslUNIQUE(im_source3, im_kmeans3)];
    Unique_gmm = [Unique_gmm, mslUNIQUE(im_source3, im_gmm3)];

    % Calculate Summer score of the three algorithms
    Summer_bilinear = [Summer_bilinear, SUMMER(im_source3, im_bilinear3)];
    Summer_kmeans = [Summer_kmeans, SUMMER(im_source3, im_kmeans3)];
    Summer_gmm = [Summer_gmm, SUMMER(im_source3, im_gmm3)];
    
    % Calculate CW-SSIM score of the three algorithms
    CWSSIM_bilinear = [CWSSIM_bilinear, cwssim_index(im_source, im_bilinear, 4, 8, 0, 0)];
    CWSSIM_kmeans = [CWSSIM_kmeans, cwssim_index(im_source, im_kmeans, 4, 8, 0, 0)];
    CWSSIM_gmm = [CWSSIM_gmm, cwssim_index(im_source, im_gmm, 4, 8, 0, 0)];
end

%%

% Save the scores of the IQA algorithms into separate excel sheets
PSNR_filename = "white_PSNR.xlsx";
PSNR_headers = {'white_PSNR_bilinear'; 'white_PSNR_kmeans'; 'white_PSNR_gmm'};
PSNR_data = [PSNR_bilinear; PSNR_kmeans; PSNR_gmm];
xlswrite(PSNR_filename, PSNR_headers, 'Sheet1', 'A1');
xlswrite(PSNR_filename, PSNR_data, 'Sheet1', 'B1');

SSIM_filename = "white_SSIM.xlsx";
SSIM_headers = {'white_SSIM_bilinear'; 'white_SSIM_kmeans'; 'white_SSIM_gmm'};
SSIM_data = [SSIM_bilinear; SSIM_kmeans; SSIM_gmm];
xlswrite(SSIM_filename, SSIM_headers, 'Sheet1', 'A1');
xlswrite(SSIM_filename, SSIM_data, 'Sheet1', 'B1');

MSUnique_filename = "white_MSUnique.xlsx";
MSUnique_headers = {'white_MSUnique_bilinear'; 'white_MSUnique_kmeans'; 'white_MSUnique_gmm'};
MSUnique_data = [MSUnique_bilinear; MSUnique_kmeans; MSUnique_gmm];
xlswrite(MSUnique_filename, MSUnique_headers, 'Sheet1', 'A1');
xlswrite(MSUnique_filename, MSUnique_data, 'Sheet1', 'B1');

CWSSIM_filename = "white_CWSSIM.xlsx";
CWSSIM_headers = {'white_CWSSIM_bilinear'; 'white_CWSSIM_kmeans'; 'white_CWSSIM_gmm'};
CWSSIM_data = [CWSSIM_bilinear; CWSSIM_kmeans; CWSSIM_gmm];
xlswrite(CWSSIM_filename, CWSSIM_headers, 'Sheet1', 'A1');
xlswrite(CWSSIM_filename, CWSSIM_data, 'Sheet1', 'B1');

Unique_filename = "white_Unique.xlsx";
Unique_headers = {'white_Unique_bilinear'; 'white_Unique_kmeans'; 'white_Unique_gmm'};
Unique_data = [Unique_bilinear; Unique_kmeans; Unique_gmm];
xlswrite(Unique_filename, Unique_headers, 'Sheet1', 'A1');
xlswrite(Unique_filename, Unique_data, 'Sheet1', 'B1');

Summer_filename = "white_Summer.xlsx";
Summer_headers = {'white_Summer_bilinear'; 'white_Summer_kmeans'; 'white_Summer_gmm'};
Summer_data = [Summer_bilinear; Summer_kmeans; Summer_gmm];
xlswrite(Summer_filename, Summer_headers, 'Sheet1', 'A1');
xlswrite(Summer_filename, Summer_data, 'Sheet1', 'B1');

CSV_filename = "white_CSV.xlsx";
CSV_headers = {'white_CSV_bilinear'; 'white_CSV_kmeans'; 'white_CSV_gmm'};
CSV_data = [CSV_bilinear; CSV_kmeans; CSV_gmm];
xlswrite(CSV_filename, CSV_headers, 'Sheet1', 'A1');
xlswrite(CSV_filename, CSV_data, 'Sheet1', 'B1');