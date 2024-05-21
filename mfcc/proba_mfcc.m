clear all; close all; clc;

Tw = 30;           % analysis frame duration (ms)
Ts = 15;           % analysis frame shift (ms)
alpha = 0.97;      % preemphasis coefficient
R = [ 300 3700 ];  % frequency range to consider
M = 30;            % number of filterbank channels
C = 12;            % number of cepstral coefficients
L = 22;            % cepstral sine lifter parameter

% hamming window 
hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));

% Read speech samples, sampling rate and precision from file
[speech,fs]=wavread('a.wav' );

% Feature extraction (feature vectors as columns)
[MFCCs,FBEs,frames]=mfcc(speech, fs, Tw, Ts, alpha, hamming, R, M, C, L );

% Plot cepstrum over time
figure, imagesc(MFCCs );
axis( 'xy' );
xlabel( 'Frame index' );
ylabel( 'Cepstrum index' );
title( 'Mel frequency cepstrum' );

MFCC_vektor=reshape(MFCCs,1,size(MFCCs,1)*size(MFCCs,2));
figure, plot(MFCC_vektor);
