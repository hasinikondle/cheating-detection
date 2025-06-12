import numpy as np
import pyaudio
import wave
from datetime import datetime
import time
import os

frequency = 2500
duration = 1000  # Only used if you want to beep

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 2000
RECORD_SECONDS = 3  # Duration to save if suspicious audio detected

# Folder to store recordings
RECORDINGS_DIR = "recordings"
if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

def audio_detection(log_list=None):
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"[ERROR] Could not initialize audio stream: {e}")
        return

    print("Listening for audio...")

    suspicious_audio_detected = False

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            if np.max(np.abs(audio_data)) > THRESHOLD and not suspicious_audio_detected:
                suspicious_audio_detected = True
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(RECORDINGS_DIR, f"suspicious_audio_{timestamp}.wav")
                
                print(f"[INFO] Suspicious audio detected, recording to {filename}")
                
                # Record additional audio
                frames = [data]
                for _ in range(0, int(RATE / CHUNK * (RECORD_SECONDS - 1))):
                    frames.append(stream.read(CHUNK, exception_on_overflow=False))
                
                # Save to WAV file
                wf = wave.open(filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                # Optional: Save to log
                if log_list is not None:
                    log_entry = f"Suspicious audio recorded at {timestamp} to {filename}"
                    log_list.append(log_entry)

            elif np.max(np.abs(audio_data)) < THRESHOLD:
                suspicious_audio_detected = False

            time.sleep(0.1)

        except Exception as e:
            print(f"[ERROR] Audio detection error: {e}")
            break

    print("Stopping audio detection...")
    stream.stop_stream()
    stream.close()
    p.terminate()
