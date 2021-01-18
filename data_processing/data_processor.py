import matplotlib.pyplot as plt
import os
import librosa
import pathlib
import csv
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.utils import make_chunks


def create_spectrograms():
    plt.figure(figsize=(10, 10))
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    cmap = plt.get_cmap('inferno')
    for genre in genres:
        pathlib.Path(f'../data/spectrograms/{genre}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'../data/music_samples/{genre}'):
            songname = f"../data/music_samples/{genre}/{filename}"
            print(songname)
            data, samplerate = librosa.load(songname, mono=True, duration=5)
            plt.specgram(data, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', scale='dB');
            plt.axis('off')
            plt.savefig(f'../data/spectrograms/{genre}/{filename[:-3].replace(".", "")}.png')
            plt.clf()


def split_audiofiles_into_parts(chunk_len_in_ms:int):
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for genre in genres:
        pathlib.Path(f'../data/music_samples_{chunk_len_in_ms/1000}/{genre}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'../data/music_samples/{genre}'):
            songname = f"../data/music_samples/{genre}/{filename}"
            sound = AudioSegment.from_file(songname)
            chunks = make_chunks(sound, chunk_len_in_ms)
            for i, chunk in enumerate(chunks):
                # we want to skip last part which is 13ms
                if i < 10:
                    splitted_fname = filename.split(".")
                    chunk_name = f"../data/music_samples_{chunk_len_in_ms/1000}/{genre}/{splitted_fname[0]}.{splitted_fname[1]}.chunk-{i}.{splitted_fname[2]}"
                    print(f"Exporting {chunk_name}")
                    chunk.export(chunk_name, format="au")


def extract_features_from_spectrogram(duration=30):
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open(f'../data/data{duration}.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for genre in genres:
        dir_path = f'../data/music_samples/{genre}' if duration == 30 else f'../data/music_samples_{duration}/{genre}'
        for filename in os.listdir(dir_path):
            songname = f"{dir_path}/{filename}"
            data, samplingrate = librosa.load(songname, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=data, sr=samplingrate)
            spec_cent = librosa.feature.spectral_centroid(y=data, sr=samplingrate)
            spec_bw = librosa.feature.spectral_bandwidth(y=data, sr=samplingrate)
            rolloff = librosa.feature.spectral_rolloff(y=data, sr=samplingrate)
            zcr = librosa.feature.zero_crossing_rate(data)
            mfcc = librosa.feature.mfcc(y=data, sr=samplingrate)
            rmse = librosa.feature.rms(y=data)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {genre}'
            file = open(f'../data/data{duration}.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())


def get_dataset(sample_len=30) -> pd.DataFrame:
    data = pd.read_csv(f"../data/data{sample_len}.csv")
    data = data.drop(['filename'], axis=1)
    return data


# create_spectrograms()
# extract_features_from_spectrogram(duration=30)
# extract_features_from_spectrogram(duration=3.0)
# split_audiofiles_into_parts(chunk_len_in_ms=3000)
