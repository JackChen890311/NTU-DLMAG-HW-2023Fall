# NTU DLMAG HW1
R12922051 資工碩一 陳韋傑

For this work (singer classification on artist20), I use the following workflow:
 - Run Demucs on audio files to do source separation
 - Transform audio file (.wav) into mel spectrogram (.png)
 - Finetuing efficientnet v2 to do image classification

 Best valid accuracy: 0.7548 / 0.9117 (Top-1 / Top-3) 