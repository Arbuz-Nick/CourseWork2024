from moviepy.editor import VideoFileClip
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Cropping2D
from tensorflow.keras.models import Model
import numpy as np
import librosa
import os
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

def load_audio(file_path, sr=16000):
    # Загрузка аудиофайла
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def audio_to_spectrogram(audio, n_fft=1024, hop_length=512):
    # Преобразование аудиосигнала в спектрограмму
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=REF)
    return S_db

def export_audio_from_video(video_path, audio_path=None):
    # Загрузка видеофайла
    video_clip = VideoFileClip(video_path)

    # Извлечение аудио дорожки
    audio_clip = video_clip.audio

    # Сохранение аудио в файл
    audio_clip.write_audiofile(audio_path if audio_path else 'clip_sound.wav')

from scipy.ndimage import zoom

def resize_spectrogram(spectrogram, target_shape=(512, 512)):
    return zoom(spectrogram, (target_shape[0] / spectrogram.shape[0], target_shape[1] / spectrogram.shape[1]))

def prepare_spectrograms(noisy_spectrograms, clean_spectrograms, max_len=256):
    # Преобразуем спектрограммы в необходимые размеры для подачи в модель
    X = []
    y = []

    for i in tqdm(range(len(noisy_spectrograms))):
        # Предполагается, что спектрограммы уже подготовлены как 2D массивы
        noisy_spec = noisy_spectrograms[i]
        clean_spec = clean_spectrograms[i]
        
        X.append(resize_spectrogram(noisy_spec))
        y.append(resize_spectrogram(clean_spec))
    
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    # Добавляем размерность для подачи в Conv2D
    X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)
    
    return X, y

def prepare_train():
    X = []
    y = []
    files = os.listdir('data/noised_speech')
    for filename in tqdm(files):
        try:
            noised, noised_sr = load_audio(f'data/noised_speech/{filename}')
            clean, clean_sr = load_audio(f'data/clean_speech/{filename}')
            X.append(audio_to_spectrogram(noised))
            y.append(audio_to_spectrogram(clean))
        except:
            continue
    print(X[0].shape)

    return prepare_spectrograms(X, y) 

def prepare_train_test():
    X = []
    y = []
    files = os.listdir('data/noised_speech')
    for filename in tqdm(files):
        try:
            noised, noised_sr = load_audio(f'data/noised_speech/{filename}')
            clean, clean_sr = load_audio(f'data/clean_speech/{filename}')
            X.append(audio_to_spectrogram(noised))
            y.append(audio_to_spectrogram(clean))
        except:
            continue
    print(X[0].shape)
    X_ready, y_ready = prepare_spectrograms(X, y) 
    return train_test_split(X_ready, y_ready, test_size=0.2, random_state=42)

def soft_mask(S, threshold):
    return S * (S > threshold)
    
def denoise_audio(noisy_audio_file, model):
    # Загрузка и преобразование аудио в спектрограмму
    noisy_audio, sr = load_audio(noisy_audio_file)
    display(IPython.display.Audio(noisy_audio, rate=sr))
    # Преобразуем аудио в спектрограмму
    noisy_spectrogram = librosa.stft(noisy_audio)
    noisy_spectrogram_shape = noisy_spectrogram.shape
    noisy_spectrogram_resized = resize_spectrogram(noisy_spectrogram)

    # Преобразуем амплитудную спектрограмму в логарифмическую шкалу (децибелы)
    noisy_db = librosa.amplitude_to_db(np.abs(noisy_spectrogram_resized), ref=ref)

    # Подготовка данных
    input_spec = np.expand_dims(noisy_db, axis=(0, -1))
    
    # Прогон через модель
    denoised_spec = model.predict(input_spec)
    
    # Преобразование обратно в аудиосигнал
    denoised_spec = denoised_spec[0, :, :, 0]  # Убираем лишние измерения
    
    # Обратное преобразование из спектрограммы в аудиосигнал
    S_recovered = librosa.db_to_amplitude(denoised_spec, ref=ref)
    
    # Используем мягкое подавление для слабых сигналов
    S_recovered_resized = resize_spectrogram(noisy_spectrogram, noisy_spectrogram_shape)

    # Применение мягкого подавления
    S_soft_masked = soft_mask(S_recovered_resized, threshold=0.02)
    
    # Восстанавливаем аудио через Griffin-Lim
    denoised_audio = librosa.griffinlim(S_soft_masked)
    display(IPython.display.Audio(denoised_audio, rate=sr))
    
    return denoised_audio, sr

# Создание модели U-Net
def unet_model(input_size=(None, None, 1)):
    inputs = Input(input_size)
    print(inputs.shape)
    # Энкодер
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    print(conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    print(pool1.shape)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    print(conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    print(pool2.shape)

    # Боттлнек
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    print(conv3.shape)

    # Декодер
    up1 = UpSampling2D(size=(2, 2))(conv3)
    print(up1.shape)
    
    # cropped_tensor1 = Cropping2D(cropping=((0, 0), (1, 0)))(up1)
    
    merge1 = Concatenate()([conv2, up1])
    print(merge1.shape)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge1)
    print(conv4.shape)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    print(up2.shape)
    
    # cropped_tensor2 = Cropping2D(cropping=((0, 0), (1, 0)))(up2)
    
    merge2 = Concatenate()([conv1, up2])
    print(merge2.shape)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
    print(conv5.shape)

    # Выходной слой
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    print(outputs.shape)

    # Создание модели
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Создание и компиляция модели
model = unet_model(input_size=(512, 512, 1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Пример тренировки (перед использованием нужно подготовить данные X_train, y_train)
# model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val))


X_train, y_train = prepare_train()
model.fit(X_train, y_train, epochs=5, batch_size=8, validation_split=0.2)

