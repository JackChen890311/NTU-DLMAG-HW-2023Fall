import os
import tqdm
import torch
import shutil
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as TT
import librosa
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import demucs.separate


with open('../data_path/train.txt') as f:
    train = f.read().split()
train = [i.split(',') for i in train]

with open('../data_path/validation.txt') as f:
    valid = f.read().split()
valid = [i.split(',') for i in valid]
test = os.listdir('artist20_testing_data')

resultPath = 'separated/mdx_extra/'
trainPath = 'separated/train/'
validPath = 'separated/valid/'
testPath = 'separated/test/'


def moveFiles(tracklist, basePath):
    for track in tqdm.tqdm(tracklist):
        [absPath, singer, song] = track
        if not os.path.exists(os.path.join(basePath, singer)):
            os.mkdir(os.path.join(basePath, singer))
        vocal = os.path.join(resultPath, song, 'vocals.mp3')
        nonVocal = os.path.join(resultPath, song, 'no_vocals.mp3')
        # yV, sr = librosa.load(vocal, sr=44100)
        # yNV, sr = librosa.load(nonVocal, sr=44100)
        # with open(os.path.join(basePath, singer, 'V_'+song+'.npy'), 'wb') as f:
        #     np.save(f, yV)
        # with open(os.path.join(basePath, singer, 'NV_'+song+'.npy'), 'wb') as f:
        #     np.save(f, yNV)
        shutil.copy(vocal, os.path.join(basePath, singer, 'V_'+song+'.mp3'))
        shutil.copy(nonVocal, os.path.join(basePath, singer, 'NV_'+song+'.mp3'))


def moveFilesTest(tracklist, basePath):
    for track in tqdm.tqdm(tracklist):
        vocal = os.path.join(resultPath, track[:-4], 'vocals.mp3')
        nonVocal = os.path.join(resultPath, track[:-4], 'no_vocals.mp3')
        shutil.copy(vocal, os.path.join(basePath, 'V_'+track[:-4]+'.mp3'))
        shutil.copy(nonVocal, os.path.join(basePath, 'NV_'+track[:-4]+'.mp3'))


if __name__ == '__main__':
    ''' Train & Valid '''
    # for l in [train, valid]:
    #     for track in tqdm.tqdm(l):
    #         if track[2] not in os.listdir(resultPath):
    #             print(track[2])
    #             demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", track[0]])

    # if not os.path.exists(trainPath):
    #     os.mkdir(trainPath)
    # if not os.path.exists(validPath):
    #     os.mkdir(validPath)

    # moveFiles(train, trainPath)
    # moveFiles(valid, validPath)

    # ORIGIN_SR = 44100
    # TARGET_SR = 16000
    # # Define transform
    # resample = T.Resample(ORIGIN_SR, TARGET_SR)
    # spectrogram = T.MelSpectrogram(sample_rate=TARGET_SR, n_fft=2048, win_length=2048, hop_length=1024)
    # resize = TT.Resize((256,512))

    # for pathUsed in [trainPath, validPath]:
    #     for label in tqdm.tqdm(os.listdir(pathUsed)):
    #         if label == 'PKS':
    #             continue
    #         result = torch.zeros((0,2,256,512))
    #         for track in os.listdir(os.path.join(pathUsed,label)):
    #             # print(track)
    #             if track[:2] != 'V_':
    #                 continue

    #             # Load audio
    #             SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(os.path.join(pathUsed,label,track))
    #             MUSIC_WAVEFORM, SAMPLE_RATE_M = torchaudio.load(os.path.join(pathUsed,label,'N'+track))

    #             SPEECH_WAVEFORM = resample(SPEECH_WAVEFORM)
    #             MUSIC_WAVEFORM = resample(MUSIC_WAVEFORM)
    #             N = SPEECH_WAVEFORM.shape[1]
                
    #             specList = []
    #             for i in range(0, N//TARGET_SR, 10):

    #                 # Perform transform
    #                 spec = librosa.power_to_db(spectrogram(SPEECH_WAVEFORM[0,i*TARGET_SR:(i+20)*TARGET_SR]))
    #                 specM = librosa.power_to_db(spectrogram(MUSIC_WAVEFORM[0,i*TARGET_SR:(i+20)*TARGET_SR]))
    #                 # print(spec.shape)

    #                 S = sum(sum(spec))
    #                 if S < -1000000 or spec.shape != (128,313):
    #                     continue
                    
    #                 specTensor = torch.tensor(spec).unsqueeze(0)
    #                 specTensor = resize(specTensor)

    #                 specMTensor = torch.tensor(specM).unsqueeze(0)
    #                 specMTensor = resize(specMTensor)

    #                 specTensorAll = torch.cat((specTensor.unsqueeze(0),specMTensor.unsqueeze(0)), dim=1)
    #                 specList.append(specTensorAll)

    #                 # print(specTensorAll.shape)

    #                 # plt.title(S)
    #                 # plt.imshow(specTensor.squeeze(), origin="lower")
    #                 # plt.axis('off')
    #                 # plt.savefig('spec/%s_%d.png'%(track,i))

    #             result = torch.cat([result] + specList)
    #             # break
    #         # break

    #         print(os.path.join(pathUsed,label), result.shape)
    #         if not os.path.exists(os.path.join(pathUsed,'PKS')):
    #             os.mkdir(os.path.join(pathUsed,'PKS'))

    #         with open(os.path.join(pathUsed,'PKS',label+'.pk'), 'wb') as f:
    #             pk.dump(result,f)


    ''' Test '''
    # for track in tqdm.tqdm(test):
    #     demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", "artist20_testing_data/"+track])
    
    # if not os.path.exists(testPath):
    #     os.mkdir(testPath)
    # moveFilesTest(test, testPath)

    ORIGIN_SR = 44100
    TARGET_SR = 16000
    # Define transform
    resample = T.Resample(ORIGIN_SR, TARGET_SR)
    spectrogram = T.MelSpectrogram(sample_rate=TARGET_SR, n_fft=2048, win_length=2048, hop_length=1024)
    resize = TT.Resize((256,512))

    pathUsed = testPath
    for track in tqdm.tqdm(os.listdir(pathUsed)):
        idx = str(int(track[-8:-4]))
        result = torch.zeros((0,2,256,512))
        if track[:2] != 'V_':
            continue

        # Load audio
        SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(os.path.join(pathUsed,track))
        MUSIC_WAVEFORM, SAMPLE_RATE_M = torchaudio.load(os.path.join(pathUsed,'N'+track))

        SPEECH_WAVEFORM = resample(SPEECH_WAVEFORM)
        MUSIC_WAVEFORM = resample(MUSIC_WAVEFORM)
        N = SPEECH_WAVEFORM.shape[1]
        
        specList = []
        for i in range(0, N//TARGET_SR, 10):

            # Perform transform
            spec = librosa.power_to_db(spectrogram(SPEECH_WAVEFORM[0,i*TARGET_SR:(i+20)*TARGET_SR]))
            specM = librosa.power_to_db(spectrogram(MUSIC_WAVEFORM[0,i*TARGET_SR:(i+20)*TARGET_SR]))
            # print(spec.shape)

            S = sum(sum(spec))
            if spec.shape != (128,313):
                continue
            
            specTensor = torch.tensor(spec).unsqueeze(0)
            specTensor = resize(specTensor)

            specMTensor = torch.tensor(specM).unsqueeze(0)
            specMTensor = resize(specMTensor)

            specTensorAll = torch.cat((specTensor.unsqueeze(0),specMTensor.unsqueeze(0)), dim=1)
            specList.append(specTensorAll)

            # print(specTensorAll.shape)

            # plt.title(S)
            # plt.imshow(specTensor.squeeze(), origin="lower")
            # plt.axis('off')
            # plt.savefig('spec/%s_%d.png'%(track,i))

        result = torch.cat(specList)

        print(os.path.join(pathUsed,idx), result.shape)

        with open(os.path.join(pathUsed,idx+'.pk'), 'wb') as f:
            pk.dump(result,f)