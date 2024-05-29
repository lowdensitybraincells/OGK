clc; clear; close all;
addpath('..\\');
addpath('..\\mfcc');
addpath('..\\Functions');


file_format = '..\\Data\\broj_%d_%d.wav';

m = 5;                          % number of distinct words
n = 265;                        % number of recordings of each word
N = 15;                         % number of windows w/o overlap
overlap = 0.5;                  % overlap (%)
N_ov = round((N-1)/overlap);    % number of windos without overlap to use
C = 12;                          % cepstral coefficient count (3, 6, 12)

% Formation of MFCC vector set
MFCCvs = zeros(C*N_ov, m*n);
labels = zeros(m, m*n);
for i = 1:m
    for j = 1:n
        file_path = sprintf(file_format,i,j);
        % Formation of the MFCC vector array
        MFCCvs(:,(i-1)*n+j) = get_MFCC_vec(file_path, N, N_ov, overlap, 'C', C);
        labels(i,(i-1)*n+j) = 1;
    end
end

X = MFCCvs;
y = labels;

net = get_network(X, y);

%% TESTING
file_format = '..\\Data_alt\\broj_%d_%d.wav';
n = 205;
numbers=1:5;
correct = 0;
labels = zeros(n*m,1);
pred = zeros(n*m,1);
for i = 1:m
    for j = 1:n
        file_path = sprintf(file_format,i,j);
        [x, fs] = audioread(file_path);
        x = x/max(abs(x));
        word = predict_word(net, x, numbers, N, N_ov, overlap, C);
        correct = correct + int32(word==i);
        labels((i-1)*n+j) = i;
        pred((i-1)*n+j) = word;

    end
end
correct*100/m/n

confusionchart(labels, pred);
