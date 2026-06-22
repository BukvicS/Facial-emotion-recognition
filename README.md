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

