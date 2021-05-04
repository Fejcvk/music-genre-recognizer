import matplotlib.pyplot as plt
import os
import librosa
import pathlib
import csv
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.utils import make_chunks
import copy


class DataProvider:
    def __init__(self,
                 input_dir: str,
                 genres: [str],
                 features: [str],
                 split_songs: bool = False,
                 splitted_song_duration_in_ms: int = None,
                 visualize: bool = None,
                 song_duration: float = 30
                 ):
        self.input_dir = input_dir
        self.genres = genres
        self.features = features
        self.split_songs = split_songs
        self.splitted_song_duration_in_ms = splitted_song_duration_in_ms
        self.visualize = visualize
        self.song_duration = song_duration
        self.audiofiles_dir_name = "/audio_files/original"
        self.audiofiles_path = self.input_dir + self.audiofiles_dir_name
        self.spectorgram_path = self.input_dir + '/spectrograms/original'
        self.output_dir = None
        self.features = ['filename'] + self.features

    def create_output_dir(self, input_dir_path: str) -> str:
        pathlib.Path(f'{input_dir_path}/../output').mkdir(parents=True, exist_ok=True)
        return f'{input_dir_path}/../output'

    def get_or_create_dataset(self) -> pd.DataFrame:
        self.output_dir = self.create_output_dir(self.input_dir)
        if self.split_songs:
            self.song_duration = int(self.splitted_song_duration_in_ms / 1000)
            self.spectorgram_path = f'{self.input_dir}/spectrograms/splitted/{self.song_duration}'
        if not pathlib.Path(f'{self.output_dir}/data_{self.song_duration}s.csv').exists():
            if self.split_songs:
                self.audiofiles_path = self.split_audiofiles_into_parts(self.splitted_song_duration_in_ms)
            self.create_spectorgrams()
            self.extract_features_from_spectrogram(duration=self.song_duration)
        return self.get_dataframe_for_csv(file_path=f'{self.output_dir}/data_{self.song_duration}s.csv')

    def create_spectorgrams(self):

        pathlib.Path(f"{self.spectorgram_path}").mkdir(parents=True, exist_ok=True)
        self.spectorgram_path = f'{self.spectorgram_path}'

        plt.figure(figsize=(10, 10))
        cmap = plt.get_cmap('inferno')
        for genre in self.genres:
            original_genre_dir_path = f'{self.audiofiles_path}/{genre}'
            pathlib.Path(f'{self.spectorgram_path}/{genre}').mkdir(parents=True, exist_ok=True)
            for filename in os.listdir(original_genre_dir_path):
                path_to_song = f'{original_genre_dir_path}/{filename}'
                if self.visualize:
                    print(path_to_song)
                data, samplerate = librosa.load(path_to_song, mono=True, duration=self.song_duration)
                plt.specgram(data, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', scale='dB');
                plt.axis('off')
                plt.savefig(f'{self.spectorgram_path}/{genre}/{filename[:-3].replace(".", "")}.png')
                plt.clf()

    def split_audiofiles_into_parts(self, chunk_len_in_ms: int) -> str:
        path_to_splitted_dataset = f'{self.input_dir}/audio_files/splitted/{chunk_len_in_ms / 1000}'
        for genre in self.genres:
            pathlib.Path(f'{path_to_splitted_dataset}/{genre}').mkdir(parents=True, exist_ok=True)
            for filename in os.listdir(f'{self.audiofiles_path}/{genre}'):
                path_to_original_song = f'{self.audiofiles_path}/{genre}/{filename}'
                sound = AudioSegment.from_file(path_to_original_song)
                chunks = make_chunks(sound, chunk_len_in_ms)
                for i, chunk in enumerate(chunks):
                    # we want to skip last part which is 13ms
                    if i < 10:
                        splitted_fname = filename.split(".")
                        chunk_name = f"{path_to_splitted_dataset}/{genre}/{splitted_fname[0]}.{splitted_fname[1]}.chunk-{i}.{splitted_fname[2]}"
                        if self.visualize:
                            print(f"Exporting {chunk_name}")
                        chunk.export(chunk_name, format="au")
        return path_to_splitted_dataset

    def extract_features_from_spectrogram(self, duration: float):
        header = copy.deepcopy(self.features)
        for i in range(1, 21):
            header.append(f' mfcc{i}')
        header.append('label')

        path_to_csv_file = f'{self.output_dir}/data_{duration}s.csv'
        csv_file = open(path_to_csv_file, 'w', newline='')
        with csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

            for genre in self.genres:
                csv_data = []
                for filename in os.listdir(f'{self.audiofiles_path}/{genre}'):
                    songname = f"{self.audiofiles_path}/{genre}/{filename}"
                    data, samplingrate = librosa.load(songname, mono=True, duration=30)
                    chroma_stft = librosa.feature.chroma_stft(y=data, sr=samplingrate)
                    spec_cent = librosa.feature.spectral_centroid(y=data, sr=samplingrate)
                    spec_bw = librosa.feature.spectral_bandwidth(y=data, sr=samplingrate)
                    rolloff = librosa.feature.spectral_rolloff(y=data, sr=samplingrate)
                    zcr = librosa.feature.zero_crossing_rate(data)
                    mfcc = librosa.feature.mfcc(y=data, sr=samplingrate)
                    rmse = librosa.feature.rms(y=data)
                    data_row = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                    for e in mfcc:
                        data_row += f' {np.mean(e)}'
                    data_row += f' {genre}'

                    csv_data.append(data_row)

                if self.visualize:
                    print(f'Saving {genre} data')

                for row in csv_data:
                    writer.writerow(row.split())

    def get_dataframe_for_csv(self, file_path: str) -> pd.DataFrame:
        data = pd.read_csv(file_path)
        data = data.drop(['filename'], axis=1)
        return data
