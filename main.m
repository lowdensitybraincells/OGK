%% Sabiranje i oduzimanje na osnovu prepoznavanja govora

clc; clear; close all;
addpath('mfcc');
addpath('Functions');

numbers = 1:9;
symbols = {'plus', 'minus', 'equals'};
fs = 22050;
samples_per_frame=1024;
MAX_TIME = 5; 
MAX_SIZE = MAX_TIME * fs;
%% MFCC extraction and quantization (optional)

file_format = 'Data\\broj_%d_%d.wav';

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
%% Neural network creation + training

% set training variables
X = MFCCvs;
y = labels;

net = get_network(X, y);

% predict_word(net, X(:, 651), numbers)

%% Record audio in real time

deviceReader = audioDeviceReader(fs, samples_per_frame);

signal = zeros(MAX_SIZE, 1);
wc = 0;

st_duration = 0.5;
th_0 = 0.002;
word_begin = 0;
SIG_DETECT = 0;
estimation_time = 3;
threshold_level = 2;

%% get signal
j = 1;
tic;

% threshold = estimate_noise(deviceReader, estimation_time, threshold_level, fs);
threshold = th_0;
% threshold = max(th_0, threshold); 

words = zeros(100, 1);

disp("Begin Signal Input...");
while true
    % get a frame
    frame = deviceReader();

    % adds frame to the signal
    frame_jdx = (j-1)*samples_per_frame+1:j*samples_per_frame;
    signal(frame_jdx) = frame;
    frame_rms = rms(frame);
    % extracting the whole signal -- can be removed
    % whole_signal(frame_idx) = frame;
    % j=j+1;

    if SIG_DETECT
        SIG_DETECT = 0;

        % detect where the word starts
        idx = detectSpeech(signal, fs);
            
        % record extra frames
        for k = 1:2
            j = j + 1;
            frame = deviceReader();
            frame_jdx = (j-1)*samples_per_frame+1:j*samples_per_frame;
            signal(frame_jdx) = frame;
        end

        % extract the word signal
        idx_1 = max(idx(end-1) - 1*samples_per_frame, 1);
        idx_2 = j*samples_per_frame;
        word_sig = signal(idx_1:idx_2);

        % set the new threshold to maximum of the minimum value, new
        % value or the average
        recorded_signal = signal(1:idx(1));
        new_threshold = threshold_level*rms(recorded_signal);
        avg_threshold = mean([threshold, new_threshold])
        threshold = max([th_0, avg_threshold, new_threshold]);

        % start recording for the next word
        signal = zeros(size(signal));
        j = 1;
        
        % predict the current word
        sound(word_sig, fs)
        clc
        word = predict_word(net, word_sig, numbers, N, N_ov, overlap, C)
        % if word == 9
        %     break;
        % end

        % if the last word is equals, finish recording
        % if strcmp(word,'9')
        %     break;
        % end
        continue;

    elseif frame_rms > threshold
        word_begin = 1;

    elseif word_begin == 1 && frame_rms < threshold
        word_begin = 0;
        SIG_DETECT = 1;

    end

    % go to the next frame
    j = j + 1;
end

figure, plot(rms_frame); 
hold on, plot(rms_sig);

% figure, plot(word_sig);
% title("extracted word")

figure, plot(linspace(0, MAX_TIME, length(signal)), signal)
disp("End Signal Input");

figure, plot(linspace(0, MAX_TIME, length(whole_signal)), whole_signal)

release(deviceReader);
detectSpeech(whole_signal, fs);