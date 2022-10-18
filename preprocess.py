"""310 infer preprocess"""
import os
import sys
import tqdm
import soundfile
import numpy as np
import librosa
import scipy.signal


def audio_to_melspectrogram(audio,
                            sample_rate=16000,
                            n_mels=80,
                            hop_length=160,
                            win_length=400,
                            n_fft=512):
    """
    Compute melspectrogram
    """
    spectrogram = librosa.feature.melspectrogram(y=audio,
                                                 sr=sample_rate,
                                                 n_mels=n_mels,
                                                 hop_length=hop_length,
                                                 win_length=win_length,
                                                 n_fft=n_fft,
                                                 window=scipy.signal.windows.hamming,
                                                 fmin=20,
                                                 fmax=7600)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def saveData(eval_list, eval_path, output_path):
    """
    Convert dataset to binary file
    """
    files = []
    labels = []
    lines = open(eval_list).read().splitlines()
    for line in lines:
        files.append(line.split()[1])
        files.append(line.split()[2])
    setfiles = list(set(files))
    setfiles.sort()
    data_path = output_path + "data/"
    label_path = output_path + "label/"

    for _, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
        name = file[0:file.find('.')]
        audio, _ = soundfile.read(os.path.join(eval_path, file))
        length = 1200 * 160 - 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_frame = np.int64(np.random.random() * (audio.shape[0] - length))
        audio = audio[start_frame : start_frame + length]
        data = np.stack([audio], axis=0)
        data = audio_to_melspectrogram(data)
        data = data - np.mean(data, axis=-1, keepdims=True)
        name = name.replace('/', '_')
        dataname = os.path.join(data_path, name + ".bin")
        data.tofile(dataname)
    for line in lines:
        labels.append(int(line.split()[0]))
    labels = np.asarray(labels)
    labelsname = os.path.join(label_path, "labels.bin")
    labels.tofile(labelsname)


if __name__ == "__main__":
    path = sys.argv[1]
    eval_data_list = os.path.join(path, "eval.txt")
    eval_data_path = os.path.join(path + "/eval/", "wav")
    output = "testdata/"
    saveData(eval_data_list, eval_data_path, output)
