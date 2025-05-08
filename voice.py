import os
import queue
import json
import struct
import sounddevice as sd
import pyaudio
from vosk import Model, KaldiRecognizer
from ollama import Client
import pvporcupine

# Paths
piper_model = "/home/leo/piper/en_US-lessac-medium.onnx"
vosk_model_path = "/home/leo/voice_ai_env/vosk-model-small-en-us-0.15"
output_path = "/home/leo/voice_ai_env/audio/output.wav"

# Ensure audio output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Initialize clients and models
ollama_client = Client(host='http://localhost:11434')

if not os.path.exists(vosk_model_path):
    print("Please download a Vosk model and place it in the specified folder.")
    exit(1)

vosk_model = Model(vosk_model_path)
samplerate = 16000
audio_queue = queue.Queue()

# Audio callback for Vosk
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(bytes(indata))

# Speech recognition
def recognize_speech():
    print("Listening... (speak a sentence)")
    recognizer = KaldiRecognizer(vosk_model, samplerate)
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                return result.get("text", "")

# Generate response using Ollama
def generate_response(prompt):
    response = ollama_client.chat(
        model='qwen2.5:0.5b',
        messages=[
            {
                'role': 'system',
                'content': (
                    "You are LEO, a sharp-tongued, audio and vision-based humanoid robot. "
                    "You reply with savage wit and brutal honesty. "
                    "Your responses are always short, confident, and sugar-coated."
                )
            },
            {
                'role': 'user',
                'content': f"Stay in character as LEO. Respond to this: {prompt}"
            }
        ]
    )
    return response['message']['content']

# Text-to-Speech using Piper
def speak(text):
    os.system(f'echo "{text}" | piper --model {piper_model} --output_file {output_path}')
    os.system(f'aplay {output_path}')

# Main loop with wake word detection
def main():
    # Setup Porcupine with keyword "vision.ppn"
    porcupine = pvporcupine.create(
        keyword_paths=["/home/leo/voice_ai_env/vision.ppn"]
    )
    
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=porcupine.sample_rate,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("Waiting for wake word... (say 'Vision')")

    try:
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            if porcupine.process(pcm) >= 0:
                print("Wake word detected!")
                speak("Yes?")
                prompt = recognize_speech()
                if prompt:
                    print(f"You said: {prompt}")
                    response = generate_response(prompt)
                    print(f"LEO: {response}")
                    speak(response)
                else:
                    speak("Sorry, I didn't catch that.")
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        stream.stop_stream()
        stream.close()
        porcupine.delete()
        pa.terminate()

if __name__ == "__main__":
    main()
