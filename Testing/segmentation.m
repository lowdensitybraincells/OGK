clc; clear; close all;
addpath('..\\');
addpath('..\\mfcc');
addpath('..\\Functions');
load("test_net.mat");

% bunch of params
wc = 0;
fs = 22050;
samples_per_frame=1024; % orig 1024 + 2 frames added
MAX_TIME = 5; 
MAX_SIZE = MAX_TIME * fs;
st_duration = 0.5;
th_0 = 0.002;
word_begin = 0;
SIG_DETECT = 0;
estimation_time = 3;
threshold_level = 2;
i = 1;
threshold = 0.01;
numbers = 1:5;
N = 15;                         % number of windows w/o overlap
overlap = 0.5;                  % overlap (%)
N_ov = round((N-1)/overlap);    % number of windos without overlap to use
C = 12;                          % cepstral coefficient count (3, 6, 12)

[x, fs] = audioread("numbers.wav");
x = x/max(abs(x));


segments = zeros(size(x));
signal = zeros(size(x));
rms_per_frame = zeros(size(x));
th_per_frame = zeros(size(x));
sig_detect_frame = zeros(size(x));
word_begin_frame = zeros(size(x)); 

N_wins = floor(length(x)/samples_per_frame);
j = 1;
words = [];

while i < N_wins
    % gets a frame
    % adds frame to the signal
    frame_idx = (i-1)*samples_per_frame+1:i*samples_per_frame;
    frame_jdx = (j-1)*samples_per_frame+1:j*samples_per_frame;
        
    frame = x(frame_idx);
    frame_rms = rms(frame);
    rms_per_frame(frame_idx) = frame_rms;
    th_per_frame(frame_idx) = threshold;
    signal(frame_jdx) = frame;
    sig_detect_frame(frame_idx) = 0.5*SIG_DETECT;
    word_begin_frame(frame_idx) = 0.75*word_begin;

    if SIG_DETECT
        SIG_DETECT = 0;
        
        % get word indices
        idx = detectSpeech(signal, fs);
        for k = 1:2
            % record extra 2 frames
            i = i + 1;
            j = j + 1;
            frame_idx = (i-1)*samples_per_frame+1:i*samples_per_frame;
            frame_jdx = (j-1)*samples_per_frame+1:j*samples_per_frame;
            frame = x(frame_idx);
            frame_rms = rms(frame);
            rms_per_frame(frame_idx) = frame_rms;
            th_per_frame(frame_idx) = threshold;
            signal(frame_jdx) = frame;
            sig_detect_frame(frame_idx) = 0.5*SIG_DETECT;
            word_begin_frame(frame_idx) = 0.75*word_begin;
        end


        % extract the word signal
        idx_1 = max(idx(end-1) - 1*samples_per_frame, 1);
        % idx_1 = idx(end-1);
        idx_2 = j*samples_per_frame;
        word_sig = signal(idx_1:idx_2);

        % set the new threshold to maximum of the minimum value, new
        % value or the average
        recorded_signal = signal(1:idx(1));
        new_threshold = threshold_level*rms(recorded_signal);
        avg_threshold = mean([threshold, new_threshold]);
        threshold = max([th_0, avg_threshold, new_threshold]);
        
        % record segment
        idx_1 = (i-j)*samples_per_frame + idx_1;
        idx_2 = (i-j)*samples_per_frame + idx_2;

        segments(idx_1:idx_2) = 0.25;

        % start recording for the next word
        signal = zeros(size(x));
        j = 1;
        i = i + 1;

        % detect word
        clc;
        word = predict_word(net, word_sig, numbers, N, N_ov, overlap, C)
        playblocking(audioplayer(word_sig, fs))
        words = [words word];
        pause(0.5)
        continue;

    elseif frame_rms > threshold
        word_begin = 1;

    elseif word_begin == 1 && frame_rms < threshold
        word_begin = 0;
        SIG_DETECT = 1;

    end
    
    % go to the next frame
    j = j + 1;
    i = i + 1;
end

% plotting the signal
t = linspace(0, length(x)/fs, length(x));
figure, plot(t, x);
hold on, plot(t, segments, 'LineWidth',2.5, 'LineStyle','-');
hold on, plot(t, rms_per_frame, 'LineWidth',2);
hold on, plot(t, th_per_frame, 'LineWidth',2);
hold on, plot(t, word_begin_frame, 'LineWidth',2);
hold on, plot(t, sig_detect_frame, 'LineWidth',2);
axis tight;

legend("x", "segments", "rms", "threshold", "WORD\_BEGIN", "SIG\_DETECT");
