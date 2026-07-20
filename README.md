# Emotion Recognition Patterns

Neuronska mreza (CNN), model za prepoznavanje 7 osnovnih emocija sa slika lica.

## Klase/emocije:

Model prepoznaje 7 emocija:
- Angry     -> Ljutnja
- Disgust   -> Gađenje
- Fear      -> Strah
- Happy     -> Sreća
- Neutral   -> Neutralno
- Sad       -> Tuga
- Surprise  -> Iznenađenje

## Arhitektura modela

- Ulaz: 48x48 slike, grayscale
- 3 layera (32 → 64 → 128 filtera)
- MaxPooling nakon svakog bloka
- Dense sloj: 128 neurona (ReLU aktivaciona funkcija)
- Izlaz: 7 neurona

Ostvarena tačnost od 65%. Rezultati prikazani na matrici konfuzije testnog skupa.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4f539387-e214-4a74-9086-0142787caee9"   width="450" alt="confusion_matrix">
</p>
