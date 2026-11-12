# Odessa Mae — generador de videos

Convierte MP3 + fotos fijas en videos listos para YouTube:

- **Sencillos** 1920×1080 con cartel inicial, letra sincronizada quemada y `.srt` aparte.
- **Shorts** 1080×1920 con el primer estribillo (30–45 s), recorte vertical centrado en la cara.
- **Compilación** con un cartel por canción y la lista de capítulos para la descripción.

Todo sale de `odessa.py` y se ajusta en `config.toml`.

## Estructura

```text
odessa-mae/
  config.toml          rutas, canciones, fuentes, tamaños, zoom, grano, audio
  odessa.py            el script
  Letras.md            letras (de acá salen los subtítulos)
  Canal.md             textos del canal (referencia, el script no lo usa)
  input/mp3/           NN-archivo-base.mp3
  input/images/        foto-NN.png (blanco y negro); color/ guarda las versiones a color
  input/fonts/         Abril Fatface y Libre Baskerville (OFL) + licencias
  input/models/        detector de rostros YuNet (OpenCV Zoo, Apache 2.0)
  work/                caché: transcripciones, alineación, audio normalizado, .ass
  output/              videos, .srt, shorts/, compilación y timestamps
```

`input/mp3`, `input/images`, `work` y `output` no van a git (son pesados).

## Requisitos

Python 3.11+, ffmpeg con libx264 y libass, y:

```bash
pip install faster-whisper opencv-python-headless rapidfuzz
```

La primera transcripción baja el modelo `medium.en` de Hugging Face (~1,5 GB).

## Uso

Desde esta carpeta:

```bash
python3 odessa.py all                    # sencillos + shorts + compilación + verificación
python3 odessa.py align                  # solo transcribe/alinea y escribe output/NN-*.srt
python3 odessa.py preview --song 1       # adelanto de 43 s del cap. 1 + su Short
python3 odessa.py singles --only 3,5     # rehacer algunos sencillos
python3 odessa.py shorts
python3 odessa.py compilation
python3 odessa.py verify                 # duración video vs. audio y volumen medido
python3 odessa.py report                 # escribe output/REPORTE.md
```

### Corregir la letra a mano

1. Editá `output/NN-archivo-base.srt` (tiempos o texto; `<i>…</i>` para cursiva).
2. Volvé a renderizar con esos tiempos:

```bash
python3 odessa.py singles --only 4 --use-srt
```

```bash
python3 odessa.py shorts --only 4 --use-srt
```

Con `--use-srt` el `.srt` no se sobrescribe y el Short toma la ventana del estribillo de ese archivo.

## Cómo funciona

- **Audio:** `loudnorm` en dos pasadas a -14 LUFS / -1 dBTP, modo lineal (solo ganancia). En la
  compilación cada canción se normaliza primero y después el conjunto completo.
- **Imagen:** la foto se pasa a blanco y negro, se escala a 4K y `zoompan` hace un zoom continuo de
  100 % a 106 % durante toda la canción. Encima, grano animado (`noise`) y viñeta suave.
- **Letra:** faster-whisper transcribe con marcas de tiempo por palabra; cada palabra de la letra se
  empareja con la transcripción por alineación global con coincidencia aproximada (las respuestas
  del coro entre paréntesis son opcionales). Se muestra siempre el texto de la letra. Cada línea
  aparece 0,2 s antes de cantarse y se va cuando empieza la siguiente o tras 1,5 s de silencio.
  Las líneas que no se pueden ubicar se interpolan entre sus vecinas y se informan.
- **Shorts:** desde 1 s antes de la primera línea del primer estribillo (etiqueta `[Chorus]` del .md)
  hasta 1 s después de su última línea; si dura menos de 30 s se extiende con las líneas siguientes,
  si pasa de 45 s se corta en una línea.

## Próximas tandas

1. Copiá los MP3 como `input/mp3/NN-archivo-base.mp3` y las fotos como `input/images/foto-NN.png`.
2. Sumá las letras al `.md` con el mismo formato (`### N. Título`, `**Lyrics**` y un bloque
   ```` ```text ```` con etiquetas `[Verse]`, `[Chorus]`…). También acepta bloques `LYRICS` … `CREDITS`.
3. Agregá una entrada `[[songs]]` por canción en `config.toml` y, si hace falta, cambiá
   `[compilation] output_name`.
4. `python3 odessa.py preview --song N` para revisar y después `python3 odessa.py all`.
