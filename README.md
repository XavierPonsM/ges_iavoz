# On-Device Keyword Spotting on ESP32-S3

> Bachelor's thesis firmware — *Artificial Intelligence to the End-Device* (UPC · ETSETB, 2023 · grade 9.5/10)

Real-time voice-command recognition running entirely on a microcontroller. Audio is captured from a microphone, converted to spectral features, and classified by a quantized **MobileNetV2** network with **TensorFlow Lite for Microcontrollers** — no cloud, no external memory for inference.

The target use case is a voice-controlled wall switch for building automation, where false activations must be close to zero.

## Results

Measured on the ESP32-S3 with the final MobileNetV2 model (keywords: *marvin, happy, eight, house* + *unknown*):

| Metric | Value |
|---|---|
| Validation accuracy | **97.18 %** |
| Test accuracy (confusion matrix) | 94.48 % |
| False positives (20 h continuous speech) | **0.05 / h** |
| Inference time | 299 ms (requirement < 500 ms) |

For comparison, the initial CNN approach on a Spanish dataset produced 13.4 false positives per hour; switching to Google Speech Commands brought it to 0.2 / h, and the MobileNetV2 transfer-learning model to 0.05 / h.

## Architecture

```
Microphone ─► Audio provider ─► Feature provider ─► TFLite Micro ─► Command recognizer ─► Callback
   (I2S)       (ring buffer)     (log-mel + VAD)     (MobileNetV2,    (posterior averaging,
                                                      int8)            thresholding)
```

| Module | Role |
|---|---|
| `ges_iavoz_audio_provider` | Captures audio into a ring buffer (FreeRTOS task) |
| `ges_iavoz_feature_provider` | Computes log-mel filterbank features with the TFLite Micro audio frontend over sliding windows |
| `ges_iavoz_main` | Builds the TFLite Micro interpreter and op resolver, runs inference |
| `ges_iavoz_command_recognizer` | Averages posteriors over time and decides when a keyword was heard |
| `ges_iavoz_command_responder` | Fires the user callback with the detected keyword |
| `libfvad` | Voice-activity detection, so inference only runs when someone is speaking |
| `esp-nn` | Espressif's optimized NN kernels for the ESP32-S3 vector extensions |

## Pipeline

1. **Data** — Google Speech Commands (35 classes, 2,000–3,000+ samples per class), curated to remove incomplete utterances.
2. **Training** — MobileNetV2 with transfer learning, based on the Edge Impulse training scripts (learning rate 0.0025, dropout 0.15, 150 epochs). Training code: [tfg-kws-training](https://github.com/XavierPonsM/tfg-kws-training).
3. **Optimization** — full-integer (int8) post-training quantization, which per TensorFlow's documentation gives ≈4× smaller models and up to ≈3× faster inference.
4. **Deployment** — model exported as a C array and compiled into the firmware.

## Hardware

- ESP32-S3-DevKitC-1U (8 MB flash). The project started on the original ESP32, but its 520 KB of SRAM was too tight for MobileNet.
- External microphone.

## Build & flash

Requires [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/) v4.4+.

```bash
git clone --recursive https://github.com/XavierPonsM/ges_iavoz.git
cd ges_iavoz
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

Detected keywords are logged on the serial monitor:

```
I (12345) Main: Received 3 with 18234, total positives: 1
```

## Limitations and future work

- Precision drops with distance and reverberation: the model tends to classify far-field speech as *unknown*. Training with noise and room-impulse-response augmentation is the next step.
- Keywords played through computer speakers were detected less reliably (77/100) than live speech (30/30).

## Credits

Developed with [Eudald Sans](https://github.com/EudaldSans) (firmware base) under the supervision of Prof. Josep Paradells (UPC). Built on TensorFlow Lite Micro's *micro_speech* example, Espressif's `esp-nn` and `libfvad`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
