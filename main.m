%% Sabiranje i oduzimanje na osnovu prepoznavanja govora

clc; clear; close all;
addpath('mfcc');
addpath('Functions');
%% MFCC extraction and quantization (optional)

file_format = 'Data\\broj_%d_%d.wav';

m = 5;                          % number of distinct words
n = 265;                        % number of recordings of each word
N = 15;                         % number of windows w/o overlap
overlap = 0.5;                  % overlap (%)
N_ov = round((N-1)/overlap);    % number of windos without overlap to use
C = 6;                          % cepstral coefficient count (3, 6, 12)

% Formation of MFCC vector set
MFCCvs = zeros(C*N_ov, m*n);
labels = zeros(m, m*n);
for i = 1:m
    for j = 1:n
        file_path = sprintf(file_format,i,j);
        % Formation of the MFCC vector array
        MFCCvs(:,(i-1)*n+j) = get_MFCC_vec(file_path, N, N_ov, overlap, C);
        labels(i,(i-1)*n+j) = 1;
    end
end

% Label set

%% Neural network creation + training

% define hidden layer sizes
hidden_layer_sizes = [25, 15, 10];

% set training variables
X = MFCCvs;
y = labels;

% create a classification network for given layeer sizes
net = patternnet(hidden_layer_sizes);

% set network options
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;
net.trainParam.max_fail = 10;

% train the network
[net, tr] = train(net, X, y, 'ShowResources', 'no');

trainInd = tr.trainInd;
testInd = tr.testInd;
valInd = tr.valInd;

% calculate accuracy on the test set 
X_test = X(:, tr.testInd);

pred = net(X_test);
predicted_labels = vec2ind(pred);
test_labels = vec2ind(labels(:, tr.testInd));

accuracy = nnz(predicted_labels == test_labels)/length(pred);

fprintf("Accuracy of number prediction is: %.2f%% \n", accuracy*100);

confusionchart(test_labels, predicted_labels);
% correctness_rate = correct/(m*n)^2;